# pvtools 并发压测

## 一次性安装（Mac Terminal）
```bash
cd ~/pvtools/loadtest          # 把 pvtools_loadtest.py 放在这里
python3 -m pip install playwright
python3 -m playwright install chromium
```

## 运行（10 人 × 10 分钟，线上）
```bash
python3 pvtools_loadtest.py
```
常用参数：
- `--users 5 --minutes 3`   先小规模试一下
- `--show`                  显示浏览器窗口，看它在点什么
- `--large-mb 40`           "大文件"上传的大小（默认 40 MB）
- `--base http://127.0.0.1:8050`  测本地

第一次运行会先生成 3 个合成上传文件（约 5 MB / 23 MB / 40 MB，放在 `upload_files/`），之后复用。

## 输出
`results/<时间>/`
- `events.jsonl`  每一步一行：用户、场景、步骤、秒数、成功/失败、错误
- `summary.csv`   每个步骤的次数 / 成功率 / p50 / p95 / 最大值
- `personas.json` 每个模拟用户的 IP、国家、设备、cookie
- `shots/`        每个失败步骤的整页截图

跑完后请把 `results/<时间>/` 这个文件夹，加上 admin 页面 Download → Events (CSV) 下载的日志一起发我。

## 注意
- 模拟用户的 cookie id 以 `lt` 开头，之后在 admin 里可以按这个前缀区分 / 排除测试流量。
- 会真实调用 OpenAI（列识别、AI diagnostic、AI coding、help chat、PV Image、Field Degradation chat），按你的要求不限次数。
- 中途按 Ctrl+C 可停止，已记录的结果会保留（summary 只在正常结束时生成）。
