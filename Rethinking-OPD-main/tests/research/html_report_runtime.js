// Exercise the report application's context/view logic. Plotly and DOM are test
// doubles; numeric data and the application JS are the actual generated report.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const payload=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const source=fs.readFileSync(process.argv[3],'utf8');
class Element {
  constructor(id=''){this.id=id;this.children=[];this.value='';this.textContent='';this.hidden=false;
    this.classList={toggle(){}};this.dataset={};this._html='';}
  set innerHTML(value){this._html=value;this.children=[];this.value='';}
  get innerHTML(){return this._html;}
  append(child){this.children.push(child);if(this.children.length===1)this.value=String(child.value);}
  addEventListener(){}
  click(){if(this.onclick)this.onclick();}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
const tabs=['metrics','overview','gallery','coverage'].map(tab=>{const e=new Element();e.dataset.tab=tab;return e;});
const plots=[];
const draw=(id,traces,layout)=>{assert(Array.isArray(traces));plots.push({id,traces,layout});return Promise.resolve();};
const sandbox={window:{REPORT_DATA:payload},document:{getElementById:get,createElement:()=>new Element(),
  querySelectorAll:q=>q==='[data-tab]'?tabs:q==='.pane'?tabs.map(t=>get(t.dataset.tab+'Pane')):
    q==='.story-plot'?[...elements.values()].filter(e=>e.id.startsWith('story')&&!e.id.startsWith('storyNote')):[],
  querySelector:q=>tabs.find(t=>q.includes('"'+t.dataset.tab+'"'))},
  Plotly:{react:draw,newPlot:draw,Plots:{resize(){}},downloadImage(){}},
  history:{replaceState(){}},location:{hash:''},Blob,URL,setTimeout,console};
vm.createContext(sandbox);
const run=code=>vm.runInContext(code,sandbox);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  vm.runInContext(source,sandbox);
  for(let i=0;i<50&&!sandbox.window.REPORT_READY;i++)await tick();
  assert(sandbox.window.REPORT_READY,'Overview should finish without JS errors');
  assert(plots.filter(p=>p.id.startsWith('story')).length>=11,'Include OTA, gap and direct geometry panels');
  for(const id of ['analysis::student_entropy_nats','analysis::rsa_rho_student_teacher','analysis::rsa_rho_to_initial']){
    const entry=payload.metrics.find(e=>e.id===id);if(!entry)continue;
    run('selectMetric(ALL.find(e=>e.id==='+JSON.stringify(id)+'))');await tick();
    if(id.includes('student_entropy')){
      assert(get('notes').textContent.includes(entry.variants[0].dims.distribution_scope==='all_valid_tokens'?
        '按有效token加权':'历史结果仅统计选定回答位置'));
    }
    if(id==='analysis::rsa_rho_student_teacher'){
      assert(get('notes').textContent.includes('同一步的S_s与T_s'));
      get('view').value='heatmap';run('render()');await tick();
      assert(plots.at(-1).traces[0].type==='heatmap');
    }
    if(id==='analysis::rsa_rho_to_initial'){
      const teacher=entry.variants.findIndex(v=>v.dims.role==='privileged_teacher');
      if(teacher>=0){get('context').value=String(teacher);run('setupContext()');await tick();
        assert(get('notes').textContent.includes('比较T_s与T_0'));}
    }
  }
  console.log('Report initialization, metric contexts and heatmap controls passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
