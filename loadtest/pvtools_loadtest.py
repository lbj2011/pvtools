#!/usr/bin/env python3
"""
pvtools concurrent load test
============================

Simulates N people using pvtools at the same time, each in its own browser
context with a different IP (X-Forwarded-For), country (CF-IPCountry),
browser / device (User-Agent + screen size), language / time zone and
visitor cookie. Every virtual user keeps picking realistic scenarios until
the time is up:

  * PV-Copilot Simple   – pick an example, YOY or PVPRO, then (sometimes)
                          AI diagnostic, AI coding (+ verify), help chat,
                          a trip to Advanced mode and back
  * PV-Copilot Advanced – prescreen -> filter (sometimes re-filter) ->
                          degradation (random methods or PVPRO) -> AI
                          diagnostic / code generation
  * PV-Copilot upload   – upload a synthetic CSV (small / medium / LARGE)
                          and run Simple analysis on it
  * Other tools         – Field Degradation (+ AI chat), PV Image (example
                          + LLM analysis), String Length Calculator, PV
                          Climate Stressors, PV Pathway, IV correction, Home

Every step is timed in the browser and written to results/<time>/:
  events.jsonl   one line per step (user, scenario, step, seconds, ok, error)
  summary.csv    per-step count / ok % / p50 / p95 / max
  shots/         a screenshot of every failed step

Test visitors get a cookie id starting with "lt" so their traffic can be
recognised (and excluded) in the admin dashboard afterwards.

Setup (once, on your Mac):
    python3 -m pip install playwright
    python3 -m playwright install chromium

Run:
    python3 pvtools_loadtest.py                      # 10 users, 10 min, production
    python3 pvtools_loadtest.py --users 5 --minutes 3
    python3 pvtools_loadtest.py --base http://127.0.0.1:8050 --show   # local, visible browsers
"""
import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import string
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit("Playwright is missing. Run:\n  python3 -m pip install playwright\n"
             "  python3 -m playwright install chromium")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Personas: where the "people" are and what they browse with
# ---------------------------------------------------------------------------
PLACES = [  # country, a public IP block in that country, time zone, locale
    ("US", "128.32", "America/Los_Angeles", "en-US"),
    ("US", "18.9", "America/New_York", "en-US"),
    ("DE", "141.20", "Europe/Berlin", "de-DE"),
    ("CN", "202.120", "Asia/Shanghai", "zh-CN"),
    ("IN", "14.139", "Asia/Kolkata", "en-IN"),
    ("FR", "193.54", "Europe/Paris", "fr-FR"),
    ("JP", "133.11", "Asia/Tokyo", "ja-JP"),
    ("AU", "130.56", "Australia/Sydney", "en-AU"),
    ("BR", "143.107", "America/Sao_Paulo", "pt-BR"),
    ("IT", "131.114", "Europe/Rome", "it-IT"),
    ("CY", "194.42", "Asia/Nicosia", "el-GR"),
    ("ES", "147.96", "Europe/Madrid", "es-ES"),
    ("KR", "147.46", "Asia/Seoul", "ko-KR"),
    ("NL", "131.155", "Europe/Amsterdam", "nl-NL"),
]
DEVICES = [  # user agent, viewport, is_mobile
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/129.0.0.0 Safari/537.36", (1440, 900), False),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/129.0.0.0 Safari/537.36", (1920, 1080), False),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0", (1536, 864), False),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.6 Safari/605.1.15", (1512, 982), False),
    ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
     (1366, 768), False),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like "
     "Gecko) Version/17.6 Mobile/15E148 Safari/604.1", (390, 844), True),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/129.0.0.0 Mobile Safari/537.36", (412, 915), True),
    ("Mozilla/5.0 (iPad; CPU OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.6 Mobile/15E148 Safari/604.1", (820, 1180), True),
]
CHAT_QUESTIONS = [
    "What's my degradation rate?",
    "Which degradation method should I try?",
    "How were points filtered?",
    "What does PVPRO add?",
    "Why is my result different from the YoY default?",
    "Is 3 years of data enough for a reliable rate?",
]
FIELD_QUESTIONS = [
    "Which climate zone shows the highest degradation?",
    "What is the median degradation rate for thin-film systems?",
    "Show systems in Germany with more than 5 years of data",
]


def make_persona(i):
    cc, block, tz, loc = random.choice(PLACES)
    ua, vp, mobile = random.choice(DEVICES)
    return {
        "user": f"u{i:02d}",
        "country": cc,
        "ip": f"{block}.{random.randint(1, 254)}.{random.randint(1, 254)}",
        "tz": tz, "locale": loc, "ua": ua, "viewport": vp, "mobile": mobile,
        # "lt" prefix = load test, so the admin dashboard can tell these apart
        "vid": "lt" + "".join(random.choices("0123456789abcdef", k=14)),
    }


# ---------------------------------------------------------------------------
# Synthetic upload files (no numpy needed)
# ---------------------------------------------------------------------------
def make_csv(path, years, step_min, seed):
    """A plausible PV time series: clear-sky-ish irradiance with clouds,
    module temperature, AC power with ~-0.6 %/yr degradation and noise."""
    if os.path.exists(path):
        return path
    rnd = random.Random(seed)
    start = datetime(2016, 1, 1, tzinfo=timezone.utc)
    n = int(years * 365 * 24 * 60 / step_min)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "ac_power_kw", "poa_irradiance_w_m2",
                    "module_temperature_c", "ambient_temperature_c"])
        cloud = 1.0
        for k in range(n):
            t = start + timedelta(minutes=k * step_min)
            doy = t.timetuple().tm_yday
            hour = t.hour + t.minute / 60
            day = math.sin(math.pi * (hour - 6) / 12) if 6 < hour < 18 else 0.0
            season = 0.75 + 0.25 * math.cos(2 * math.pi * (doy - 172) / 365)
            cloud = min(1.0, max(0.15, cloud + rnd.gauss(0, 0.05)))
            poa = max(0.0, 1050 * day * season * cloud + rnd.gauss(0, 8)) if day else 0.0
            amb = 15 + 10 * math.cos(2 * math.pi * (doy - 200) / 365) + 6 * day + rnd.gauss(0, 1)
            tmod = amb + poa * 0.03 + rnd.gauss(0, 0.8)
            age = k * step_min / (365 * 24 * 60)
            p = poa / 1000 * 100 * (1 - 0.004 * (tmod - 25)) * (1 - 0.006 * age)
            p = max(0.0, p + rnd.gauss(0, 0.6)) if poa > 0 else 0.0
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), f"{p:.3f}", f"{poa:.1f}",
                        f"{tmod:.2f}", f"{amb:.2f}"])
    return path


def prepare_uploads(folder, large_mb):
    os.makedirs(folder, exist_ok=True)
    files = {
        "small": make_csv(os.path.join(folder, "upload_small_15min_3yr.csv"), 3, 15, 1),
        "medium": make_csv(os.path.join(folder, "upload_medium_5min_5yr.csv"), 5, 5, 2),
    }
    # LARGE: 1-min data; years chosen so the file is roughly large_mb
    yrs = max(1.0, large_mb / 37.0)          # ~37 MB per year of 1-min rows
    files["large"] = make_csv(os.path.join(folder, f"upload_large_1min_{yrs:.1f}yr.csv"),
                              yrs, 1, 3)
    for k, p in files.items():
        print(f"  upload file {k:6s}: {os.path.getsize(p) / 1e6:6.1f} MB  {os.path.basename(p)}")
    return files


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, out):
        self.out = out
        os.makedirs(os.path.join(out, "shots"), exist_ok=True)
        self.f = open(os.path.join(out, "events.jsonl"), "a")
        self.rows = []
        self.t0 = time.time()

    def add(self, **row):
        row["t"] = round(time.time() - self.t0, 1)
        row["wall"] = datetime.now().isoformat(timespec="seconds")
        self.rows.append(row)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        mark = "ok " if row.get("ok") else "ERR"
        print(f"[{row['t']:7.1f}s] {row['user']} {mark} {row['scenario']:<18} "
              f"{row['step']:<28} {row.get('secs', 0):7.1f}s "
              f"{(row.get('error') or row.get('note') or '')[:70]}", flush=True)

    def summary(self):
        by = {}
        for r in self.rows:
            by.setdefault((r["scenario"], r["step"]), []).append(r)
        lines = []
        for (sc, st), rs in sorted(by.items()):
            secs = sorted(r["secs"] for r in rs if r.get("ok") and r.get("secs") is not None)
            ok = sum(1 for r in rs if r.get("ok"))

            def pct(q):
                return round(secs[min(len(secs) - 1, int(q * len(secs)))], 1) if secs else None
            lines.append({"scenario": sc, "step": st, "count": len(rs),
                          "ok_pct": round(100 * ok / len(rs)),
                          "p50_s": round(statistics.median(secs), 1) if secs else None,
                          "p95_s": pct(0.95), "max_s": round(max(secs), 1) if secs else None})
        with open(os.path.join(self.out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lines[0].keys()) if lines else ["scenario"])
            w.writeheader()
            w.writerows(lines)
        return lines


class StepFailed(Exception):
    pass


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------
def sel_id(i):
    """CSS selector for any element id (Dash pattern ids contain JSON)."""
    return f'[id={json.dumps(i)}]'


MODE_TAB = lambda m: sel_id(json.dumps({"mode": m, "type": "mode-tab"}, separators=(",", ":")))
ADV_STEP = lambda n: sel_id(json.dumps({"step": n, "type": "advanced-step-tab"},
                                       separators=(",", ":")))


class VU:
    """One virtual user."""

    def __init__(self, persona, base, rec, uploads, deadline, think):
        self.p, self.base, self.rec, self.uploads = persona, base.rstrip("/"), rec, uploads
        self.deadline, self.think = deadline, think
        self.page = None
        self.scenario = ""
        self.last_action = ""
        self.example = None

    # -- timing ---------------------------------------------------------------
    async def step(self, name, coro, note=None):
        t = time.perf_counter()
        try:
            res = await coro
            self.rec.add(user=self.p["user"], scenario=self.scenario, step=name,
                         secs=round(time.perf_counter() - t, 2), ok=True,
                         note=note if note is not None else (res if isinstance(res, str) else None),
                         country=self.p["country"], ip=self.p["ip"])
            return res
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            shot = os.path.join(self.rec.out, "shots",
                                f"{self.p['user']}_{int(time.time())}_{name}.png")
            try:
                await self.page.screenshot(path=shot, full_page=True)
            except Exception:
                shot = None
            self.rec.add(user=self.p["user"], scenario=self.scenario, step=name,
                         secs=round(time.perf_counter() - t, 2), ok=False, error=err,
                         shot=shot, country=self.p["country"], ip=self.p["ip"])
            raise StepFailed(err)

    async def pause(self, a=None, b=None):
        a, b = a or self.think[0], b or self.think[1]
        await asyncio.sleep(random.uniform(a, b))

    # -- page helpers ---------------------------------------------------------
    async def goto(self, path, ready_sel, timeout=90):
        await self.page.goto(self.base + path, wait_until="domcontentloaded",
                             timeout=timeout * 1000)
        await self.page.wait_for_selector(ready_sel, state="visible", timeout=timeout * 1000)
        return path

    async def click(self, sel, timeout=30):
        """Click like a person; if the control is attached but not clickable
        (carousel item, hidden radio input, overlay) fall back to a DOM click."""
        self.last_action = f"click {sel}"
        loc = self.page.locator(sel).first
        await loc.wait_for(state="attached", timeout=timeout * 1000)
        try:
            await loc.scroll_into_view_if_needed(timeout=5000)
            await loc.click(timeout=8000)
        except PWTimeout:
            await loc.evaluate("e => e.click()")

    async def wait_js(self, js, timeout, arg=None):
        await self.page.wait_for_function(js, arg=arg, timeout=timeout * 1000, polling=500)

    async def text(self, sel):
        try:
            return (await self.page.locator(sel).first.inner_text(timeout=2000)) or ""
        except Exception:
            return ""

    async def check_alert(self, sel):
        """Raise if the area shows one of the app's ERROR callouts (a red
        "⚠ ..." block or an AI error box). Advisory notes don't count."""
        msg = await self.page.evaluate(r"""(sel) => {
            const a = document.querySelector(sel); if (!a) return null;
            const box = a.querySelector('.pvc-ai-diagnostic-error');
            if (box && box.innerText.trim()) return box.innerText;
            const hard = a.innerText.match(/(No data points survived[^\n]{0,160}|Please (click|apply|upload|run)[^\n]{0,160}|Couldn't identify the required columns[^\n]{0,120})/);
            if (hard) return hard[0];
            const m = a.innerText.match(/⚠\s*([^\n]{0,240})/);
            if (m && /error|failed|could ?n[o']t|unavailable|lost|timed? ?out|longer than expected|not found|please/i.test(m[1]))
                return m[0];
            return null; }""", sel)
        if msg:
            raise StepFailed("app error: " + " ".join(msg.split())[:200])

    # ======================================================================
    # PV-Copilot
    # ======================================================================
    async def copilot_open(self):
        await self.step("open page", self.goto("/pv-copilot", "#simple-analyze-btn"))

    # examples that carry DC voltage + current (needed by PVPRO)
    PVPRO_EXAMPLES = (1, 2, 6, 7)

    async def pick_example(self, for_pvpro=False):
        n = random.choice(self.PVPRO_EXAMPLES) if for_pvpro else random.randint(1, 9)
        self.example = n
        await self.click("#example-tab-list")

        async def _load():
            await self.click(f"#load-example-btn-{n}")
            await self.wait_js("""() => { const b = document.querySelector('#simple-analyze-btn');
                return b && !b.disabled && !/Upload/.test(b.innerText); }""", 120)
            return f"example {n}"
        await self.step("load example", _load())
        return n

    async def simple_run(self, method, timeout):
        await self.click(f'#simple-method-radio label:has(input[value="{method}"])')

        async def _run():
            await self.click("#simple-analyze-btn")
            await self.wait_js("""() => {
                const v = document.querySelector('#simple-result-view');
                const s = document.querySelector('#simple-status');
                const done = v && v.offsetParent !== null && v.innerText.length > 40;
                const err = s && /⚠|Error|failed|Couldn't|Could not/.test(s.innerText);
                return done || err; }""", timeout)
            await self.check_alert("#simple-status")
            return method
        await self.step(f"simple analyze {method}", _run())

    async def ai_diagnostic(self, prefix):
        async def _run():
            await self.click(f"#{prefix}-ai-diagnostic-btn")
            await self.wait_js(f"""() => {{ const o = document.querySelector('#{prefix}-ai-diagnostic-output');
                return o && o.innerText.trim().length > 20; }}""", 180)
            await self.check_alert(f"#{prefix}-ai-diagnostic-output")
        await self.step(f"AI diagnostic ({prefix})", _run())

    async def ai_code(self, prefix, verify):
        btn = "#simple-code-btn" if prefix == "simple" else "#generate-code-btn"
        out = "#simple-code-output" if prefix == "simple" else "#code-preview"
        vbtn = "#simple-code-verify-btn" if prefix == "simple" else "#code-verify-btn"
        vout = "#simple-code-verify-output" if prefix == "simple" else "#code-verify-output"

        async def _gen():
            await self.click(btn)
            await self.wait_js("""([o, b]) => { const e = document.querySelector(o), k = document.querySelector(b);
                return e && e.innerText.trim().length > 80 && k && !k.disabled; }""", 420,
                               arg=[out, btn])
            await self.check_alert(out)
        await self.step(f"AI coding: generate ({prefix})", _gen())
        if verify:
            async def _ver():
                await self.click(vbtn, timeout=20)
                await self.wait_js("""([o, b]) => { const e = document.querySelector(o), k = document.querySelector(b);
                    return e && /verified|reproduc|differ|match|error|Failed|✓|✗/i.test(e.innerText) && k && !k.disabled; }""",
                                   420, arg=[vout, vbtn])
            await self.step(f"AI coding: verify ({prefix})", _ver())

    async def help_chat(self):
        q = random.choice(CHAT_QUESTIONS)

        async def _ask():
            if not await self.page.locator("#chat-composer").first.is_visible():
                await self.click("#chat-drawer-open")
            await self.page.locator("#chat-composer").first.fill(q)
            await self.click("#chat-send")
            await self.wait_js("""(q) => { const h = document.querySelector('#chat-history');
                if (!h) return false; const t = h.innerText, i = t.lastIndexOf(q);
                return i >= 0 && t.slice(i + q.length).trim().length > 20; }""", 180, arg=q)
            return q
        await self.step("AI help chat", _ask())
        try:
            await self.click("#chat-drawer-close", timeout=5)
        except Exception:
            pass

    async def sc_copilot_simple(self):
        self.scenario = "copilot-simple"
        await self.copilot_open()
        await self.pause(2, 6)
        method = random.choices(["YOY", "PVPRO"], weights=[3, 1])[0]
        await self.pick_example(for_pvpro=method == "PVPRO")
        await self.pause(1, 4)
        await self.simple_run(method, 900 if method == "PVPRO" else 300)
        await self.pause()
        if random.random() < 0.6:
            await self.ai_diagnostic("simple")
            await self.pause()
        if random.random() < 0.5:
            await self.ai_code("simple", verify=random.random() < 0.6)
            await self.pause()
        if random.random() < 0.5:
            await self.help_chat()
            await self.pause()
        if random.random() < 0.3:     # peek at Advanced, come back, run Simple again
            await self.step("switch to Advanced", self.click(MODE_TAB("advanced")))
            await self.pause(3, 8)
            await self.step("switch to Simple", self.click(MODE_TAB("simple")))
            await self.pause(2, 5)
            try:
                await self.click("#simple-result-return", timeout=5)
            except Exception:
                pass
            other = ("PVPRO" if method == "YOY" and self.example in self.PVPRO_EXAMPLES
                     else "YOY")
            await self.simple_run(other, 900 if other == "PVPRO" else 300)

    async def adv_prescreen(self):
        async def _run():
            await self.click("#analyze-btn")
            await self.wait_js("""() => { const b = document.querySelector('#analyze-btn');
                const o = document.querySelector('#data-summary-output');
                return b && !b.disabled && /prescreening/i.test(b.innerText) && o && o.innerText.length > 50; }""",
                               300)
            await self.check_alert("#data-summary-output")
        await self.step("adv prescreen", _run())

    async def adv_filter(self, label="adv filter"):
        async def _run():
            await self.click("#filter-btn")
            await self.wait_js("""() => { const b = document.querySelector('#filter-btn');
                const o = document.querySelector('#data-filter-output');
                return b && !b.disabled && o && o.innerText.length > 50; }""", 300)
            await self.check_alert("#data-filter-output")
        await self.step(label, _run())

    async def adv_degradation(self):
        pvpro = getattr(self, "want_pvpro", False)
        if pvpro:
            await self.click('#metric-pvpro-radio input[value="PVPRO"]')
            label = "PVPRO"
        else:
            picks = random.sample(["LR", "HW", "ARIMA", "CSD"], k=random.randint(0, 2))
            for m in picks:
                await self.page.locator(f'#metric-stat-radio input[value="{m}"]').first.check()
            label = "+".join(["YOY"] + picks)

        async def _run():
            await self.click("#run-btn")
            await self.wait_js("""() => { const d = document.querySelector('#degradation-output');
                const p = document.querySelector('#pvpro-progress-output');
                const b = document.querySelector('#run-btn');
                const txt = (d ? d.innerText : '') + (p ? p.innerText : '');
                return b && !b.disabled && /Calculate/.test(b.innerText) && txt.length > 50; }""",
                               900 if pvpro else 300)
            await self.check_alert("#degradation-output")
            await self.check_alert("#pvpro-progress-output")
            return label
        await self.step(f"adv degradation {'PVPRO' if pvpro else 'stats'}", _run())

    async def sc_copilot_advanced(self):
        self.scenario = "copilot-advanced"
        await self.copilot_open()
        await self.pause(2, 5)
        self.want_pvpro = random.random() < 0.3
        await self.pick_example(for_pvpro=self.want_pvpro)
        await self.step("switch to Advanced", self.click(MODE_TAB("advanced")))
        await self.pause(1, 4)
        await self.adv_prescreen()
        await self.pause()
        await self.click("#advanced-next-step-1")
        await self.pause(1, 4)
        await self.adv_filter()
        if random.random() < 0.3:
            await self.pause(2, 6)
            await self.adv_filter("adv re-filter")
        await self.pause()
        await self.click("#advanced-next-step-2")
        await self.pause(1, 4)
        await self.adv_degradation()
        await self.pause()
        if random.random() < 0.5:
            await self.ai_diagnostic("advanced")
            await self.pause()
        if random.random() < 0.3:     # back to Simple for a quick run on the same data
            await self.step("switch to Simple", self.click(MODE_TAB("simple")))
            await self.pause(2, 5)
            await self.simple_run("YOY", 300)
            await self.step("switch to Advanced", self.click(MODE_TAB("advanced")))
            await self.pause(2, 5)
        if random.random() < 0.4:
            await self.click(ADV_STEP(4))
            await self.pause(1, 3)
            await self.ai_code("advanced", verify=random.random() < 0.4)
        if random.random() < 0.3:
            await self.help_chat()

    async def sc_copilot_upload(self):
        self.scenario = "copilot-upload"
        size = random.choices(["small", "medium", "large"], weights=[3, 2, 1])[0]
        path = self.uploads[size]
        await self.copilot_open()
        await self.pause(2, 6)

        async def _up():
            await self.page.locator("#upload-data input[type=file]").set_input_files(path)
            await self.wait_js("""() => { const b = document.querySelector('#simple-analyze-btn');
                return b && !b.disabled && !/Upload/.test(b.innerText); }""", 300)
            return f"{size} {os.path.getsize(path) / 1e6:.1f} MB"
        await self.step(f"upload {size}", _up())
        await self.pause(1, 4)
        await self.simple_run("YOY", 600)
        await self.pause()
        if random.random() < 0.4:
            await self.ai_diagnostic("simple")

    # ======================================================================
    # Other tools
    # ======================================================================
    async def sc_field(self):
        self.scenario = "field-degradation"
        await self.step("open page", self.goto("/field-degradation", "#field-map", 120))
        await self.pause(3, 8)
        if random.random() < 0.7:
            q = random.choice(FIELD_QUESTIONS)

            async def _ask():
                if not await self.page.locator("#chat-input").first.is_visible():
                    await self.click("#ai-toggle-btn")
                await self.page.locator("#chat-input").first.fill(q)
                await self.click("#chat-submit")
                await self.page.wait_for_timeout(1500)
                await self.wait_js("""() => { const b = document.querySelector('#chat-submit');
                    return b && !b.disabled; }""", 240)
                return q
            await self.step("AI chat", _ask())

    async def sc_pv_image(self):
        self.scenario = "pv-image"
        await self.step("open page", self.goto("/pv-image", "#analyze-button"))
        await self.pause(2, 5)
        await self.click(f"#example{random.randint(1, 9)}")
        await self.pause(1, 3)

        async def _an():
            await self.click("#analyze-button")
            await self.wait_js("""() => { const r = document.querySelector('#image-analysis-result');
                return r && r.innerText.trim().length > 40; }""", 240)
            await self.check_alert("#image-analysis-result")
        await self.step("LLM image analysis", _an())

    async def sc_string(self):
        self.scenario = "string-length"
        await self.step("open page", self.goto("/string-length-calculator", "#get-weather"))
        await self.pause(2, 5)

        async def _map():
            await self.page.locator("#lat").fill(f"{random.uniform(25, 48):.2f}")
            await self.page.locator("#lon").fill(f"{random.uniform(-122, -72):.2f}")
            await self.click("#get-weather")
            await self.wait_js("""() => !!document.querySelector('#map .maplibregl-canvas, #map canvas')""",
                               120)
        await self.step("show nearest location", _map())

    async def sc_climate(self):
        self.scenario = "climate-stressors"
        await self.step("open page", self.goto("/pv-climate-stressors", "#pvcz-map-select", 120))
        await self.pause(3, 8)

    async def sc_simple_visit(self, path, name, sel):
        self.scenario = name
        await self.step("open page", self.goto(path, sel, 120))
        await self.pause(3, 10)

    # ======================================================================
    async def run(self, browser):
        p = self.p
        ctx = await browser.new_context(
            user_agent=p["ua"], viewport={"width": p["viewport"][0], "height": p["viewport"][1]},
            is_mobile=p["mobile"], has_touch=p["mobile"], locale=p["locale"], timezone_id=p["tz"],
            extra_http_headers={"X-Forwarded-For": p["ip"], "CF-IPCountry": p["country"]},
        )
        host = self.base.split("://", 1)[1].split("/")[0].split(":")[0]
        await ctx.add_cookies([{"name": "pvtools_client_id", "value": p["vid"],
                                "domain": host, "path": "/"}])
        scenarios = [
            (self.sc_copilot_simple, 30), (self.sc_copilot_advanced, 22),
            (self.sc_copilot_upload, 14), (self.sc_field, 9), (self.sc_pv_image, 7),
            (self.sc_string, 6), (self.sc_climate, 4),
            (lambda: self.sc_simple_visit("/pv-pathway", "pv-pathway", "body"), 3),
            (lambda: self.sc_simple_visit("/iv-curve-correction-tool", "iv-correction", "body"), 2),
            (lambda: self.sc_simple_visit("/", "home", "body"), 3),
        ]
        while time.time() < self.deadline:
            self.page = await ctx.new_page()
            self.page.set_default_timeout(60000)
            fn = random.choices([s for s, _ in scenarios], weights=[w for _, w in scenarios])[0]
            try:
                await fn()
            except StepFailed:
                pass
            except Exception as e:     # a helper (click etc.) failed outside a timed step
                self.rec.add(user=p["user"], scenario=self.scenario or "?", step="(navigation)",
                             secs=0, ok=False,
                             error=f"{self.last_action}: {type(e).__name__}: "
                                   f"{str(e).splitlines()[0][:140]}",
                             country=p["country"], ip=p["ip"])
            finally:
                try:
                    await self.page.close()
                except Exception:
                    pass
            await self.pause(3, 12)
        await ctx.close()


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="https://pvtools.lbl.gov")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--ramp", type=float, default=60, help="seconds to spread user starts over")
    ap.add_argument("--large-mb", type=float, default=40, help="size of the LARGE upload file")
    ap.add_argument("--show", action="store_true", help="show the browser windows")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--think", type=float, nargs=2, default=(4, 15),
                    help="min/max seconds a user pauses between actions")
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    out = os.path.join(HERE, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"pvtools load test -> {a.base}  users={a.users}  minutes={a.minutes}\nresults: {out}")
    print("preparing upload files (first run only)…")
    uploads = prepare_uploads(os.path.join(HERE, "upload_files"), a.large_mb)
    rec = Recorder(out)
    personas = [make_persona(i + 1) for i in range(a.users)]
    with open(os.path.join(out, "personas.json"), "w") as f:
        json.dump(personas, f, indent=1)
    for p in personas:
        print(f"  {p['user']}  {p['country']}  {p['ip']:<16} {p['locale']:<6} "
              f"{'mobile ' if p['mobile'] else 'desktop'}  vid={p['vid']}")

    deadline = time.time() + a.ramp + a.minutes * 60
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not a.show, **({"executable_path": os.environ["PW_CHROME"]} if os.environ.get("PW_CHROME") else {}))

        async def start(i, p):
            await asyncio.sleep(i * a.ramp / max(1, a.users))
            await VU(p, a.base, rec, uploads, deadline, a.think).run(browser)
        try:
            await asyncio.gather(*(start(i, p) for i, p in enumerate(personas)))
        finally:
            await browser.close()

    lines = rec.summary()
    print("\n=== summary (seconds, measured in the browser) ===")
    print(f"{'scenario':<18} {'step':<30} {'n':>4} {'ok%':>4} {'p50':>7} {'p95':>7} {'max':>7}")
    for r in lines:
        print(f"{r['scenario']:<18} {r['step']:<30} {r['count']:>4} {r['ok_pct']:>4} "
              f"{str(r['p50_s']):>7} {str(r['p95_s']):>7} {str(r['max_s']):>7}")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped.")
