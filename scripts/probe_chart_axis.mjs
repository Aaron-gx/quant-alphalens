#!/usr/bin/env node
/**
 * 图表坐标探针：读 ECharts 实例的真实 option 与像素映射。
 *
 * 为什么需要它
 * ------------
 * 「预测标注钉在哪一天」肉眼只能看个大概。截图里那条曲线被挤在左侧一小条、
 * 右边大片留白时，光看图**无法判断**是数据长度不对、还是日期网格被撑大了、
 * 还是坐标轴 min/max 设错了。这里直接把 ECharts 内部的
 * xAxis.data 长度、各 series 的 data 长度、以及 markPoint 的
 * convertToPixel 结果打出来，一眼定位。
 *
 * 用法：node scripts/probe_chart_axis.mjs <url> [--port 9350] [--wait 13000]
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  join(process.env.LOCALAPPDATA || '', 'Google\\Chrome\\Application\\chrome.exe'),
].find((p) => p && existsSync(p));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let port = 9350, wait = 13000, url = null;
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--port') port = Number(process.argv[++i]);
  else if (process.argv[i] === '--wait') wait = Number(process.argv[++i]);
  else url = process.argv[i];
}
if (!url) { console.error('用法: node scripts/probe_chart_axis.mjs <url>'); process.exit(2); }

const PROBE = `(async () => {
  const canvases = [...document.querySelectorAll('canvas')];
  // 本项目用 \`import * as echarts from 'echarts'\`，模块名没有挂到 window。
  // Vite dev 会把裸导入预打包成 /node_modules/.vite/deps/echarts.js，
  // 直接动态 import 这个 URL 就能拿到**同一个模块实例**（含实例注册表），
  // 从而用 getInstanceByDom 读到运行时真正的 option。
  let echartsObj = window.echarts;
  const tried = [];
  if (!echartsObj || !echartsObj.getInstanceByDom) {
    for (const u of ['/node_modules/.vite/deps/echarts.js', '/@id/echarts']) {
      try {
        const m = await import(u);
        const cand = m.getInstanceByDom ? m : (m.default && m.default.getInstanceByDom ? m.default : null);
        tried.push(u + ':' + (cand ? 'ok' : 'no-getInstanceByDom'));
        if (cand) { echartsObj = cand; break; }
      } catch (e) { tried.push(u + ':ERR ' + e.message); }
    }
  }
  if (!echartsObj || !echartsObj.getInstanceByDom) {
    return JSON.stringify({ error: '无法取得 echarts 模块', tried, canvasCount: canvases.length });
  }
  const insts = [];
  for (const c of canvases) {
    const host = c.closest('div[_echarts_instance_]') || c.parentElement;
    if (!host) continue;
    const inst = echartsObj.getInstanceByDom(host);
    if (!inst) continue;
    const opt = inst.getOption();
    const xAxis = opt.xAxis || [];
    const main = xAxis[0] || {};
    const xLen = Array.isArray(main.data) ? main.data.length : null;
    insts.push({
      domId: host.id || null,
      hostW: host.clientWidth, hostH: host.clientHeight,
      xType: main.type,
      xDataLen: xLen,
      xFirst: Array.isArray(main.data) ? main.data[0] : null,
      xLast: Array.isArray(main.data) ? main.data[main.data.length - 1] : null,
      series: (opt.series || []).map((s) => ({
        name: s.name, type: s.type,
        dataLen: Array.isArray(s.data) ? s.data.length : null,
        nonNull: Array.isArray(s.data) ? s.data.filter((v) => v != null && v !== '-').length : null,
      })),
      markPoints: (opt.series || [])
        .filter((s) => s.markPoint && Array.isArray(s.markPoint.data))
        .map((s) => ({
          seriesName: s.name,
          items: s.markPoint.data.map((m) => {
            let px = null;
            try { px = inst.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, m.coord); } catch (e) { px = 'ERR:' + e.message; }
            return { coord: m.coord, value: m.value, px };
          }),
        })),
      // **真实渲染出来的文本框**（zrender 显示列表），不是估算。
      // 标注是否被截断、是否互相压字，只有这里量出来的边界矩形说了算。
      markLabelBoxes: (() => {
        const out = [];
        try {
          const list = inst.getZr().storage.getDisplayList();
          for (const el of list) {
            const st = el && el.style;
            const txt = st && typeof st.text === 'string' ? st.text : '';
            if (!/看涨|看跌|震荡|方向不明/.test(txt)) continue;
            const r = el.getBoundingRect ? el.getBoundingRect() : null;
            if (!r) continue;
            // 相对画布定位（transform 已在 rect 里算入，但需加上图元的绝对位移）
            const ax = (el.transform ? el.transform[4] : 0) || 0;
            const ay = (el.transform ? el.transform[5] : 0) || 0;
            out.push({
              text: txt,
              x: +(ax + r.x).toFixed(1), y: +(ay + r.y).toFixed(1),
              w: +r.width.toFixed(1), h: +r.height.toFixed(1),
            });
          }
        } catch (e) { return { error: e.message }; }
        // 同一段文字可能被渲染成多份（含隐藏副本），按几何去重
        const seen = {};
        return out.filter((b) => {
          // ⚠️ PROBE 是外层模板字符串，内部不能出现 \${...}（会被 Node 抢先插值），
          // 所以这里用拼接而不是模板串。
          const k = b.text + '|' + b.x + '|' + b.y + '|' + b.w;
          if (seen[k]) return false;
          seen[k] = 1; return true;
        });
      })(),
      gridPx: (() => {
        try {
          const a = inst.convertToPixel({ xAxisIndex: 0 }, 0);
          const b = inst.convertToPixel({ xAxisIndex: 0 }, xLen ? xLen - 1 : 0);
          return { firstX: a, lastX: b, span: (typeof a === 'number' && typeof b === 'number') ? b - a : null };
        } catch (e) { return 'ERR:' + e.message; }
      })(),
    });
  }
  return JSON.stringify({ moduleTried: tried, insts, canvasCount: canvases.length }, null, 1);
})()`;

async function main() {
  const profile = mkdtempSync(join(tmpdir(), 'probe-cdp-'));
  const child = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-first-run',
    '--disable-extensions', '--hide-scrollbars', '--window-size=1440,1000',
    `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'],
    { stdio: 'ignore', windowsHide: true });
  let ws = null;
  const cleanup = () => { try { ws?.close(); } catch { } try { child.kill(); } catch { }
    setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch { } }, 600); };
  try {
    const dl = Date.now() + 20000;
    let ver = null;
    while (Date.now() < dl) {
      try { const r = await fetch(`http://127.0.0.1:${port}/json/version`); if (r.ok) { ver = await r.json(); break; } } catch { }
      await sleep(250);
    }
    if (!ver) throw new Error('CDP 未就绪');
    const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
    ws = new WebSocket(t.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.addEventListener('open', res, { once: true }); ws.addEventListener('error', rej, { once: true }); });
    let id = 0; const pend = new Map();
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString());
      if (m.id && pend.has(m.id)) { const { res, rej } = pend.get(m.id); pend.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
    });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = ++id; pend.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params }));
      setTimeout(() => { if (pend.has(i)) { pend.delete(i); rej(new Error('CDP 超时 ' + method)); } }, 20000);
    });
    await send('Page.enable'); await send('Runtime.enable');
    await send('Page.navigate', { url });
    await sleep(wait);
    const r = await send('Runtime.evaluate', { expression: PROBE, returnByValue: true, awaitPromise: true });
    console.log('===PROBE===');
    const raw = r?.result?.value || JSON.stringify(r);
    console.log(raw);

    // ── 断言：标注必须落在画布内，且互不压字 ────────────────────────────────
    // 全部基于 **zrender 真实渲染出的边界矩形**（markLabelBoxes），不是估算：
    // ① 越出画布 → 界面上显示成 "一季 看涨 +0.6…"（实测踩过）
    // ② 两个矩形相交 → 后者盖住前者的字（实测踩过：「一月 看涨 +0.18%」压「次日 方向不明」）
    // 这两种缺陷看图都不容易发现（"字本来就这么长吧"），必须量化。
    try {
      const rep = JSON.parse(raw);
      let rc = 0;
      for (const inst of rep.insts || []) {
        const boxes = inst.markLabelBoxes;
        if (!Array.isArray(boxes) || !boxes.length) continue;
        console.log(`\n[画布 ${inst.hostW}×${inst.hostH}] 实测渲染出 ${boxes.length} 个方向标签：`);
        for (const b of boxes) {
          const overL = b.x < 0;
          const overR = b.x + b.w > inst.hostW;
          if (overL || overR) {
            rc = 1;
            console.log(`FAIL  溢出画布: "${b.text}" x∈[${b.x},${(b.x + b.w).toFixed(0)}] `
              + `画布宽 ${inst.hostW}`);
          } else {
            console.log(`   ok  "${b.text}" x∈[${b.x.toFixed(0)},${(b.x + b.w).toFixed(0)}] `
              + `y∈[${b.y.toFixed(0)},${(b.y + b.h).toFixed(0)}]`);
          }
        }
        let hits = 0, pairs = 0;
        for (let i = 0; i < boxes.length; i++) {
          for (let j = i + 1; j < boxes.length; j++) {
            const a = boxes[i], b = boxes[j];
            pairs++;
            const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
            const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
            if (ox > 0 && oy > 0) {
              hits++; rc = 1;
              const s = ox * oy;
              console.log(`FAIL  标注重叠: "${a.text}" ×"${b.text}" `
                + `重叠 ${ox.toFixed(0)}×${oy.toFixed(0)}px（面积 ${s.toFixed(0)}）`);
            }
          }
        }
        console.log(hits === 0
          ? `PASS  ${boxes.length} 个标签均在画布内、${pairs} 对两两不相交`
          : `结果：${hits}/${pairs} 对相交`);
      }
      process.exitCode = rc;
    } catch (e) {
      console.log('（未能解析探针输出做几何断言：' + e.message + '）');
    }
  } finally { cleanup(); }
}
main().catch((e) => { console.error('===PROBE-ERROR===', e?.stack || String(e)); process.exit(1); });
