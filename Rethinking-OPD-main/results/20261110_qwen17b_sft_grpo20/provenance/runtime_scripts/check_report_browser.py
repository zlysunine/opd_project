from pathlib import Path
import json,sys
from playwright.sync_api import sync_playwright
report=Path(sys.argv[1])
errors=[];network=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium-1187/chrome-linux/chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
 page=browser.new_page(viewport={'width':1500,'height':1100},accept_downloads=True)
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:network.append(r.url) if r.url.startswith(('https:','http:')) else None)
 page.goto(report.as_uri());page.wait_for_function('window.REPORT_READY')
 overview=page.locator('#reportTitle').inner_text()
 page.click('[data-tab="metrics"]')
 count=page.evaluate('''async () => {for(const m of ALL){selectMetric(m);await render();if(!document.getElementById('plot').data.length)throw Error(m.id);}return ALL.length;}''')
 page.locator('#search').fill('entropy');assert page.locator('#metricList button').count()>0
 page.locator('#search').fill('')
 page.evaluate("selectMetric(ALL.find(m=>m.name==='curvature_deg'&&m.kind==='analysis'))")
 page.select_option('#view','heatmap')
 page.wait_for_function("document.getElementById('plot').data[0].type==='heatmap'")
 assert page.evaluate("document.getElementById('plot').data[0].x.length")==28
 with page.expect_download() as d:page.click('#downloadCsv')
 d.value.save_as(str(report.with_suffix('.test.csv')))
 with page.expect_download() as d:page.click('#downloadPng')
 d.value.save_as(str(report.with_suffix('.test.png')))
 with page.expect_download() as d:page.click('#downloadSvg')
 d.value.save_as(str(report.with_suffix('.test.svg')))
 page.click('[data-tab="coverage"]')
 assert page.locator('#coverageTable tbody tr').count()==count
 page.screenshot(path=str(report.with_suffix('.preview.png')),full_page=False)
 assert not errors,errors
 assert not network,network
 browser.close()
result=dict(html=str(report),title=overview,rendered_metrics=count,page_errors=errors,network_requests=network,checks=['all metrics','search','28-layer heatmap','CSV','PNG','SVG','coverage table'])
report.with_suffix('.browser_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
