#!/usr/bin/env node
/**
 * 零依赖前端控制台验收脚本（Chrome DevTools Protocol）。
 *
 * 目的：验证「[ECharts] Instance ... has been disposed」警告是否已被 EChart 包装组件修复，
 * 并顺带区分 stdout 里出现的外部脚本错误（如 stadium.js）是不是本项目产物。
 *
 * 用法：
 *   node scripts/e2e_console_check.mjs <url> [--route2 <url2>] [--wait 12000] [--out out.json]
 *
 * 设计要点：
 * - 只用 Node 内置能力（node:child_process / fetch / WebSocket），不依赖 playwright。
 * - 直接复用系统已安装的 Chrome（--headless=new + CDP），无需下载浏览器。
 * - 采集 Runtime.consoleAPICalled / Runtime.exceptionThrown / Log.entryAdded 三类事件。
 * - 除首屏加载外，还会在 SPA 内做客户端路由跳转（触发组件卸载/重挂载），
 *   以复现「频繁 remount 期间实例被 dispose」的竞态场景。
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME_CANDIDATES = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  join(process.env.LOCALAPPDATA || '', 'Google\\Chrome\\Application\\chrome.exe'),
];

function parseArgs(argv) {
  const out = { url: null, routes: [], wait: 12000, out: null, keepOpen: false, port: 9333, clickTabs: false, dom: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--route2') out.routes.push(argv[++i]);
    else if (a === '--wait') out.wait = Number(argv[++i]);
    else if (a === '--out') out.out = argv[++i];
    else if (a === '--port') out.port = Number(argv[++i]);
    else if (a === '--keep') out.keepOpen = true;
    else if (a === '--click-tabs') out.clickTabs = true;
    else if (a === '--dom') out.dom = true;
    else if (a === '--no-route') out.noRoute = true;
    else if (a === '--shot') out.shot = argv[++i];
    else if (a === '--viewport-shot') out.viewportShot = true;
    else if (!out.url) out.url = a;
  }
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForCdp(port, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (r.ok) return await r.json();
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error(`CDP 端口 ${port} 在 ${timeoutMs}ms 内未就绪`);
}

/** 极简 CDP 客户端：单条 WebSocket，自动给请求分配 id。 */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.handlers = [];
    ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString()); }
      catch { return; }
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method) {
        for (const h of this.handlers) h(msg.method, msg.params, msg.sessionId);
      }
    });
  }
  static async connect(wsUrl) {
    const ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', (e) => reject(new Error(`WS 连接失败: ${e?.message || 'unknown'}`)), { once: true });
    });
    return new Cdp(ws);
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP 超时: ${method}`));
        }
      }, 20000);
    });
  }
  on(fn) { this.handlers.push(fn); }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

/** 把 CDP 的 RemoteObject 还原成可读字符串。 */
function remoteToText(arg) {
  if (!arg) return '';
  if (arg.type === 'string') return String(arg.value);
  if ('value' in arg && arg.value !== undefined) {
    return typeof arg.value === 'object' ? JSON.stringify(arg.value) : String(arg.value);
  }
  if (arg.unserializableValue) return String(arg.unserializableValue);
  if (arg.description) return String(arg.description);
  return `<${arg.type}>`;
}

function frameOf(stack) {
  const f = stack?.callFrames?.[0];
  if (!f) return '';
  return `${f.functionName || '(anon)'} @ ${(f.url || '').split('/').pop()}:${f.lineNumber + 1}`;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.url) {
    console.error('用法: node scripts/e2e_console_check.mjs <url> [--route2 <url2>] [--wait ms] [--out file]');
    process.exit(2);
  }

  const chrome = CHROME_CANDIDATES.find((p) => p && existsSync(p));
  if (!chrome) throw new Error('未找到 Chrome，请检查 CHROME_CANDIDATES');

  const profile = mkdtempSync(join(tmpdir(), 'e2e-cdp-'));
  const child = spawn(chrome, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-networking',
    '--hide-scrollbars',
    '--window-size=1440,1000',
    `--remote-debugging-port=${args.port}`,
    `--user-data-dir=${profile}`,
    'about:blank',
  ], { stdio: 'ignore', windowsHide: true });

  const consoleMsgs = [];
  const exceptions = [];
  const logEntries = [];
  let cdp = null;

  const cleanup = () => {
    try { cdp?.close(); } catch { /* ignore */ }
    try { child.kill(); } catch { /* ignore */ }
    setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ } }, 800);
  };

  try {
    const ver = await waitForCdp(args.port);
    // 新建一个 taget，拿到它的 WS 调试地址
    const tRes = await fetch(`http://127.0.0.1:${args.port}/json/new?about:blank`, { method: 'PUT' });
    const target = await tRes.json();
    cdp = await Cdp.connect(target.webSocketDebuggerUrl);

    cdp.on((method, params) => {
      if (method === 'Runtime.consoleAPICalled') {
        consoleMsgs.push({
          level: params.type,
          text: (params.args || []).map(remoteToText).join(' '),
          at: frameOf(params.stackTrace),
          ts: params.timestamp,
        });
      } else if (method === 'Runtime.exceptionThrown') {
        const d = params.exceptionDetails || {};
        exceptions.push({
          text: d.exception?.description || d.text || '(no description)',
          url: d.url || '',
          line: d.lineNumber,
          col: d.columnNumber,
          at: frameOf(d.stackTrace),
        });
      } else if (method === 'Log.entryAdded') {
        const e = params.entry || {};
        logEntries.push({ level: e.level, source: e.source, text: e.text, url: e.url || '', line: e.lineNumber });
      }
    });

    await cdp.send('Runtime.enable');
    await cdp.send('Log.enable');
    await cdp.send('Page.enable');
    await cdp.send('Network.enable');

    // ① 首屏加载
    await cdp.send('Page.navigate', { url: args.url });
    await sleep(Math.min(args.wait, 8000));

    // ② 真实点击 tab 来回切换 —— 复现「KlineChart 卸载/重挂载期间实例被 dispose」竞态。
    //    Analysis.tsx 用 <ErrorBoundary key={symbol + activeTab}> 包裹，切 tab 会让整棵
    //    子树卸载重建；只有 overview 这页才渲染 KlineChart，所以来回切 = 反复 init/dispose。
    const tabClicks = [];
    if (args.clickTabs) {
      const clickTab = async (label) => {
        const expr = `(() => {
          const tabs = [...document.querySelectorAll('[role="tablist"] button[role="tab"]')];
          const hit = tabs.find(b => (b.textContent || '').trim() === ${JSON.stringify(label)});
          if (!hit) return 'notfound';
          hit.click();
          return 'clicked';
        })()`;
        const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true });
        return r?.result?.value;
      };
      const pingpong = ['技术指标与盘口', '综合概览', '舆情与相关性', '综合概览',
                        '基本面与持仓', '综合概览', '历史预测对账', '综合概览'];
      const TAB = '综合概览';
      for (const label of pingpong) {
        tabClicks.push({ label, result: await clickTab(label) });
        // 故意用很短的间隔：异步 init 还没完成就被下一次卸载打断 —— 竞态窗口就在这里
        await sleep(350);
      }
      // 最后快速双击同一 tab，制造最极端的连续 remount
      for (let i = 0; i < 4; i++) {
        tabClicks.push({ label: TAB, result: await clickTab(i % 2 === 0 ? '技术指标与盘口' : TAB) });
        await sleep(180);
      }
    }

    // ③ SPA 内客户端路由跳转（补一层卸载/重挂载触发）
    //    ⚠️ pushState 会把 URL 里的 query（如 ?symbol=016665）冲掉，页面随即回落到默认标的，
    //    导致 --dom 断言断的是"另一个标的"的 DOM。需要在指定标的上断言时必须加 --no-route。
    //    tab 点击本身已覆盖 remount 场景，跳过这一段不会削弱控制台检查的强度。
    for (const path of (args.noRoute ? [] : ['/', '/analysis', '/verify', '/analysis', ...args.routes])) {
      const pathname = new URL(path, args.url).pathname;
      await cdp.send('Runtime.evaluate', {
        expression: `window.history.pushState({}, '', ${JSON.stringify(pathname)}); window.dispatchEvent(new PopStateEvent('popstate'));`,
        awaitPromise: false,
      });
      await sleep(1200);
    }

    // ③ 留足时间让异步 dataload / React StrictMode 二次挂载走完
    await sleep(Math.max(args.wait - 8000, 3000));

    // ④ 抓取页面里是否还挂着已 dispose 的 ECharts 实例
    const probe = await cdp.send('Runtime.evaluate', {
      expression: `(() => {
        const canvases = document.querySelectorAll('canvas');
        let huge = 0;
        canvases.forEach(c => { if (c.width > 40 && c.height > 40) huge++; });
        return JSON.stringify({ title: document.title, canvasCount: canvases.length, chartCanvases: huge, path: location.pathname, readyState: document.readyState });
      })()`,
      returnByValue: true,
    });

    const bus = ver?.Browser || '';

    // ⑤ 可选的 DOM 断言：确认关键组件真的渲染出来了，而不只是"没有报错"
    let domProbe = null;
    if (args.dom) {
      const domRes = await cdp.send('Runtime.evaluate', {
        expression: `(() => {
          const q = (s) => document.querySelector(s);
          const verdict = q('[data-testid="direction-verdict"]');
          const text = (el) => (el ? (el.innerText || el.textContent || '').replace(/\\s+\\n/g, '\\n').trim() : null);
          // "自学习已校准" 徽标：calibration.method === 'identity' 时不应出现
          const calibBadges = [...document.querySelectorAll('span')]
            .filter((s) => (s.textContent || '').trim() === '自学习已校准').length;
          return JSON.stringify({
            path: location.pathname + location.search,
            verdictPresent: !!verdict,
            verdictText: text(verdict),
            calibratedBadgeCount: calibBadges,
            asOfLine: (() => {
              const hit = [...document.querySelectorAll('div,span')]
                .find((el) => (el.textContent || '').includes('数据截至')
                  && (el.textContent || '').length < 200
                  && !el.querySelector('div'));
              return hit ? (hit.textContent || '').replace(/\\s+/g, ' ').trim() : null;
            })(),
            // 预测区的起算基准与各周期到期日。这两项必须能从 DOM 里读到 ——
            // 否则"图上的标注到底钉在哪天"就只能靠看图猜，验收不了。
            planLine: (() => {
              const hit = [...document.querySelectorAll('div,span')]
                .find((el) => (el.textContent || '').includes('预测自')
                  && (el.textContent || '').length < 400
                  && !el.querySelector('div'));
              return hit ? (hit.textContent || '').replace(/\\s+/g, ' ').trim() : null;
            })(),
            // 方向裁决卡里逐周期的「到期 MM-DD」
            verdictRowDues: [...document.querySelectorAll('[data-testid="direction-verdict"] *')]
              .map((el) => (el.textContent || '').trim())
              .filter((t) => /^到期 \\d{2}-\\d{2}$/.test(t)),
          });
        })()`,
        returnByValue: true,
      });
      try { domProbe = JSON.parse(domRes?.result?.value || 'null'); } catch { domProbe = null; }
    }

    // ⑥ 可选整页截图：把"改完长什么样"直接产出成文件给用户看，
    //    比让用户自己开浏览器对着控制台报告验收有效得多。
    let shotBytes = 0;
    if (args.shot) {
      // ⚠️ captureBeyondViewport: true 会把视口临时改成"整页高度"再截图，
      // 这个过程会触发 ResizeObserver → chart.resize()。ECharts 的 canvas
      // 若在 resize 重绘完成前被截到，就会得到一张**几何形状不对**的图
      // （实测出现过：绘图区被压成左侧 10% 宽的窄条）。
      // 所以做几何验收时必须用 --viewport-shot（只截视口内、不改视口尺寸）。
      const shot = await cdp.send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: !args.viewportShot,
      });
      const buf = Buffer.from(shot.data, 'base64');
      writeFileSync(args.shot, buf);
      shotBytes = buf.length;
    }

    const report = {
      target: args.url,
      browser: bus,
      ua: ver?.['User-Agent'] || '',
      page: probe?.result?.value ? JSON.parse(probe.result.value) : null,
      dom: domProbe,
      shot: args.shot ? { path: args.shot, bytes: shotBytes } : null,
      tabClicks,
      counts: { console: consoleMsgs.length, exceptions: exceptions.length, logs: logEntries.length },
      // 关键判定
      disposedWarnings: [...consoleMsgs, ...logEntries]
        .filter((m) => /has been disposed/i.test(m.text))
        .map((m) => ({ text: m.text, at: m.at || `${m.url}:${m.line}` })),
      echartsMsgs: consoleMsgs.filter((m) => /echarts/i.test(m.text)).map((m) => ({ level: m.level, text: m.text, at: m.at })),
      reactMsgs: consoleMsgs.filter((m) => /react/i.test(m.text)).map((m) => ({ level: m.level, text: m.text })),
      errors: consoleMsgs.filter((m) => m.level === 'error').map((m) => ({ text: m.text, at: m.at })),
      warnings: consoleMsgs.filter((m) => m.level === 'warning').map((m) => ({ text: m.text, at: m.at })),
      exceptions,
      // 外部脚本来源识别（stadium.js 等）
      externalScriptHits: logEntries.filter((e) => /stadium\.js/i.test(e.text || '') || /stadium\.js/i.test(e.url || '')),
      allConsole: consoleMsgs,
      allLogs: logEntries,
    };

    const json = JSON.stringify(report, null, 2);
    if (args.out) writeFileSync(args.out, json, 'utf8');

    console.log('===E2E-REPORT-BEGIN===');
    console.log(json);
    console.log('===E2E-REPORT-END===');
  } finally {
    if (!args.keepOpen) cleanup();
  }
}

main().catch((err) => {
  console.error('===E2E-ERROR===', err?.stack || String(err));
  process.exit(1);
});
