"""Read-only review snapshot: readable TXT + lossless, content-addressed evidence."""
from pathlib import Path, PurePosixPath
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import base64, csv, hashlib, json, re, tarfile
from datetime import datetime, timezone
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'PairGuide_CGAN_Usage_Review_20260915.txt'
LIMIT = 50_000_000
STAMP = datetime.now(timezone.utc).isoformat()
records, excluded, blobs, image_index, inputs = [], [], {}, [], {}
seen_paths = set()
sha = lambda b: hashlib.sha256(b).hexdigest()
jsonbytes = lambda x: (json.dumps(x, ensure_ascii=False, indent=2)+'\n').encode()
studies = sorted(p for p in (ROOT/'Data').glob('PairGuide*') if p.is_dir()
                 and not any(s in p.name for s in ['Aborted','ReviewBundle']))

def add(name, data, origin=None, **metadata):
    assert name not in seen_paths, name
    assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
    seen_paths.add(name)
    digest=sha(data); blobs.setdefault(digest, data)
    records.append(dict(path=name, bytes=len(data), sha256=digest, origin=origin or name, **metadata))

def add_file(p):
    data=p.read_bytes();name=str(p.relative_to(ROOT));inputs[name]=sha(data)
    add(name,data)

def omit(p, reason):
    excluded.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,reason=reason))

# Include exact runtime dependencies, public entry points, operators and local tests.
runtime=json.loads((ROOT/'Data/PairGuideVoronoiUse_20260915/source_manifest.json').read_text())['files']
for r in runtime:
    p=ROOT/r['path'];assert sha(p.read_bytes())==r['sha256'],r['path']
    if not r['path'].startswith('Data/'):add_file(p)
for folder in ['Algorithms/Multi-objective optimization/CBS-CGAN',
               'Algorithms/Multi-objective optimization/PairGuide',
               'Algorithms/Multi-objective optimization/CCMO', 'Algorithms/Utility functions']:
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file() and p.suffix in {'.m','.md','.py'} and str(p.relative_to(ROOT)) not in seen_paths:
            add_file(p)
add_file(ROOT/'Agent.md');add_file(ROOT/'Data/PairGuideUseResearch_CURRENT.json')

# Results/diagnostics stay exact. Large raw states are represented by their CSV exports;
# retain selected model states separately for mechanism design. Never select by performance.
image_jobs=[]
for study in studies:
    reports='\n'.join(p.read_text(errors='replace') for p in study.glob('*.md'))
    for p in sorted(study.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(ROOT));ext=p.suffix.lower()
        if name in seen_paths:continue
        if any(x.startswith('.') or x in {'__pycache__','_staging','review_numeric'} for x in p.relative_to(study).parts):
            omit(p,'cache or redundant expanded numeric export');continue
        if ext in {'.png','.jpg','.jpeg','.webp'}:
            if 'qa' in p.relative_to(study).parts or any(s in p.stem.lower() for s in ['contact','preview','collage']):
                omit(p,'visual QA/contact sheet; original plots retained');continue
            # Older first-use grids repeat similar views; retain report-cited figures and
            # all precomputed diagnosis/IGD overviews. All Sep8+ scientific PNGs are kept.
            old=('20260906' in study.name or '20260907' in study.name or not re.search(r'_2026\d{4}$',study.name))
            if old and not (p.name in reports or re.search(r'igd|overview|summary|diagnos',p.name,re.I)):
                omit(p,'early supplementary first-use image; reports and numerical records retained');continue
            image_jobs.append(p);continue
        if p.name.endswith('.tar.gz') and re.search(r'source|runtime|snapshot|contract',p.name,re.I):
            data=p.read_bytes();inputs[name]=sha(data)
            with tarfile.open(fileobj=BytesIO(data)) as archive:
                for member in archive.getmembers():
                    if not member.isfile():continue
                    part=PurePosixPath(member.name)
                    assert not part.is_absolute() and '..' not in part.parts
                    if part.suffix in {'.m','.py','.md','.json','.patch','.diff'}:
                        b=archive.extractfile(member).read()
                        add('SourceSnapshots/'+name+'/'+member.name,b,
                            origin=name+'::'+member.name, archiveSHA256=sha(data))
            continue
        if ext in {'.m','.py','.md','.csv','.json','.txt','.diff','.patch'}:
            is_external=(('research' in p.relative_to(study).parts and
                p.name not in {'RESEARCH.md','README.md','PLAN.md','PROTOCOL.md','METHODS_AND_LIMITS.md'})
                and ext in {'.md','.json','.txt'} and p.name not in {'p6_timepoint_review.csv'})
            if is_external or p.name.startswith(('official_','github_','search_','fetch_')):
                omit(p,'downloaded third-party source body/search response; citations retained in reports');continue
            if ext=='.json' and p.stat().st_size>2_000_000:
                omit(p,'redundant large state JSON; tabular observations retained');continue
            if ext=='.txt' and (p.stat().st_size>100_000 or re.search(r'log|stdout|stderr|console',p.name,re.I)):
                omit(p,'startup/runtime transcript; structured checks retained');continue
            add_file(p);continue
        omit(p,'raw MAT/FIG state, duplicate PDF, or execution log; numerical exports and selected checkpoints retained')

# Exact full G weights, q/Z/C and first/10k/30k/50k states used by recent usage diagnostics.
state_paths=[]
for pattern in ['Data/PairGuideCriticUse_20260915/qa/condition_state_P*.mat',
                'Data/PairGuideSideDirection_20260915/qa/side_state_P*.mat',
                'Data/PairGuideManifoldUse_20260914/qa/first_use_P*.mat',
                'Data/PairGuideManifoldUse_20260914/results/*seed01_generator_tangent.mat']:
    state_paths.extend(ROOT.glob(pattern))
for p in sorted(state_paths):
    add_file(p)
    excluded[:]=[r for r in excluded if r['path']!=str(p.relative_to(ROOT))]

cache=HERE/'image_cache';cache.mkdir(exist_ok=True)
def convert(p):
    original=p.read_bytes();srcsha=sha(original)
    with Image.open(BytesIO(original)) as src:
        # Preserve the displayed image exactly on a white figure background.
        im=Image.alpha_composite(Image.new('RGBA',src.size,'white'),src.convert('RGBA')).convert('RGB')
        c=cache/(srcsha+'.webp')
        if c.exists():data=c.read_bytes()
        else:
            f=BytesIO();im.save(f,format='WEBP',lossless=True,method=4);data=f.getvalue();c.write_bytes(data)
        with Image.open(BytesIO(data)) as check:
            check.load();assert check.size==im.size and check.convert('RGB').tobytes()==im.tobytes(),p
        return p,data,dict(originalSHA256=srcsha,originalBytes=len(original),width=im.width,height=im.height,
            encoding='lossless WebP, original dimensions; original RGBA displayed on white',
            displayedRGBSHA256=sha(im.tobytes()))

def image_priority(p):
    study=p.relative_to(ROOT).parts[1];name=p.name.lower()
    if re.search(r'_2026091[45]$',study):return 0
    if re.search(r'igd|overview|summary|search_|comparisons|mechanism_development|allocation_grid',name):return 1
    if study in {'PairGuideStageDiagnosis_20260908','PairGuideEarlyMidTuning_20260909',
                 'PairGuideMechanismValidation_20260910','PairGuideProposalCompletion_20260911',
                 'PairGuideFeasibleCause_20260910','PairGuideUseDesign_20260911'}:return 2
    if re.search(r'first|fe050000|fe100000|stage01|first1000',name) and '_full' in name:return 3
    if '_focus' in name:return 5
    return 4

image_jobs.sort(key=lambda p:(image_priority(p),str(p)))
image_bytes=0
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,(p,data,meta) in enumerate(pool.map(convert,image_jobs),1):
        if image_bytes+len(data)>12_000_000:
            assert image_priority(p)>0,'Current-contract figures must never be omitted'
            omit(p,'historical supplementary/repeated view; current-era figures, IGD and primary diagnosis figures take priority under single-TXT cap')
            continue
        image_bytes+=len(data)
        name=str(p.relative_to(ROOT));inputs[name]=meta['originalSHA256']
        target=name+'.webp';add(target,data,origin=name,**meta)
        image_index.append(dict(path=target,sourcePath=name,**meta))
        if i%50==0:print(f'IMAGE_VERIFIED {i}/{len(image_jobs)}',flush=True)

def generated(name, text):add('Review/'+name,text.encode(),origin='generated for this read-only review')

# Recompute a common current-contract table from exact per-run summary CSVs.
cases=[('bridge','PairGuideSequentialTuning_20260911',5,'results/stage3_grid/P{p}_seed{s}_cgan_z12_g8x8_d16x16_b64_i1000_r20_a050000.csv'),
       ('boundary_local','PairGuideBoundaryUse_20260914',5,'results/LIRCMOP{p}_BC_seed{s:02d}_cgan.csv'),
       ('boundary_segment','PairGuideUseSearch_20260914',5,None),('boundary_tube','PairGuideUseSearch_20260914',2,None),
       ('generator_tangent','PairGuideManifoldUse_20260914',2,None),
       ('generator_tangent_anchor','PairGuideAnchorUse_20260915',2,None),
       ('generator_tangent_anchor_energy','PairGuideAnchorUse_20260915',2,None),
       ('critic_bridge','PairGuideCriticUse_20260915',2,None),('feasible_ray','PairGuideFeasibleRay_20260915',2,None),
       ('feasible_ray_de','PairGuideRayDE_20260915',2,None),('feasible_chord','PairGuideFeasibleChord_20260915',2,None),
       ('coordinate_bridge','PairGuideCoordinateUse_20260915',2,None),('voronoi_bridge','PairGuideVoronoiUse_20260915',2,None)]
import math
fields=['AUC_200_100000','IGDfinal','AUC_200_20000','AUC_20000_50000','AUC_50000_100000']
def load_metrics(rel):
    row=next(csv.DictReader((ROOT/rel).open()))
    assert int(row.get('searchFE',row.get('FE',0)))==100000
    trace=Path(rel).with_name(Path(rel).stem+'_trajectory.csv')
    values=list(csv.DictReader((ROOT/trace).open()))
    times=[int(float(v['FE'])) for v in values];ys=[float(v['IGD']) for v in values]
    assert times==list(range(200,100001,200)) and all(math.isfinite(y) and y>0 for y in ys)
    assert math.isclose(ys[-1],float(row['IGDfinal']),rel_tol=1e-11)
    result={'IGDfinal':ys[-1]}
    for field,lo,hi in zip([fields[0],*fields[2:]],[200,200,20000,50000],[100000,20000,50000,100000]):
        val=sum((ys[i]+ys[i+1])*(times[i+1]-times[i])/2 for i in range(len(ys)-1)
                if times[i]>=lo and times[i+1]<=hi)/(hi-lo)
        if field in row:assert math.isclose(val,float(row[field]),rel_tol=1e-10),rel
        result[field]=val
    return result
summary=[];metric_rows=[]
for arm,study,n,template in cases:
    ratios={k:[] for k in fields};per={}
    for p in range(5,9):
        per[p]={k:[] for k in fields}
        for s in range(1,n+1):
            rel=f'Data/{study}/'+(template or 'results/LIRCMOP{p}_BC_seed{s:02d}_'+arm+'.csv').format(p=p,s=s)
            a=load_metrics(rel)
            de=f'Data/PairGuideBoundaryUse_20260914/results/LIRCMOP{p}_BC_seed{s:02d}_de_slots.csv'
            b=load_metrics(de)
            for k in fields:
                r=float(a[k])/float(b[k]);assert math.isfinite(r) and r>0
                ratios[k].append(r);per[p][k].append(r)
                metric_rows.append([arm,p,s,k,float(a[k]),float(b[k]),r,rel,de])
    stat=lambda v:dict(percent=100*math.expm1(sum(map(math.log,v))/len(v)),wins=sum(x<1 for x in v),pairs=len(v))
    summary.append(dict(arm=arm,study=study,seeds=list(range(1,n+1)),metrics={k:stat(v) for k,v in ratios.items()},
                        perProblem={str(p):{k:stat(v) for k,v in vv.items()} for p,vv in per.items()}))
generated('CURRENT_FULL_RUN_COMPARISONS.json',jsonbytes(summary).decode())
f=BytesIO()
import io
stream=io.StringIO();w=csv.writer(stream);w.writerow(['arm','problem','seed','metric','candidate','de_slots','ratio','candidateSource','referenceSource']);w.writerows(metric_rows)
generated('CURRENT_PAIRED_METRICS.csv',stream.getvalue())

overview='''# 当前证据总览（2026-09-15，只读打包）

目标尚未完成。当前默认仍是bridge，Q20/盒外最多4/.5截止。下表全部在当前固定参数下，每次100000FE，比较同骨架DE；负数更好。过程为200–100000FE原生P1 IGD梯形积分/99800；先算同题同种子比，再取几何平均。图的算术均值不是此配对汇总。每行保留全部对应运行，不删坏例。

5种子是开发证据，seed1–5已反复使用；2种子只是初筛，不能跨样本数将表排序成可靠冠军。bridge为已核验的历史同条件复用。DE复用不算新增独立证据。

| 用法 | 每题种子数 | 全程过程变化 | 最终变化 | 后期变化 | P6最终变化 |
|---|---:|---:|---:|---:|---:|
'''
for row in summary:
    m=row['metrics'];overview+=f"| {row['arm']} | {len(row['seeds'])} | {m[fields[0]]['percent']:+.2f}% | {m[fields[1]]['percent']:+.2f}% | {m[fields[4]]['percent']:+.2f}% | {row['perProblem']['6'][fields[1]]['percent']:+.2f}% |\n"
overview+='''
boundary_local整体过程−4.27%、最终−5.16%，但P6和后期退化，两个主指标名义95%区间均跨1；这是值得理解的局部有效机制，不是已经全面成功。boundary_segment从2种子扩至5种子后终点优势消失，P7 seed5真实发生大幅退化，保留该例。后续多数算子仍在P6或晚期失败。

## 已尝试的机制及阅读路径

- 当前bridge：p+0.5(q−p)+0.5(a−b)，接原OperatorDE。boundary_local用q定位真实F/I邻域，将桥接后的实际子代压入以(F+I)/2为中心、半径||F−I||/2的开球；boundary_segment/tube改变局部几何。精确实现见当前PairGuideBoundaryStep_RC/PairGuideBoundarySearch_RC及各历史源码。
- generator_tangent：保留G局部输出子空间内的DE分量，使用互补空间q拉力；anchor和anchor_energy改用真实F锚定/保持步长。critic_bridge加入现有D梯度。feasible_ray及feasible_ray_de以F为起点在q或q+DE方向收缩；feasible_chord用q定位两个真实F形成弦。
- coordinate_bridge把平均半强度q拉力改成私有随机流的15/30坐标完整拉力；voronoi_bridge把原桥接真实子代限制在档案F相对I的近邻半空间。两者8次完整筛选均失败；公式不能保证真实可行或接近CPF。
- 仅诊断、没有完整性能结论：LocalReflect（反射及软映射），DirectionUse（DE向q反射、角平分线），LeaderUse（q选真实F引导），TangentPull（保留完整DE+G子空间q拉力），GeneratorOrient（G子空间内反射DE的反向分量）。对应REPORT.md、诊断CSV和原型.m均收录。很多单代IGD持平，不能当成完整运行等效或无望。
- SideDirection为0FE只读模型诊断，不是已完成性能实验。G(z,w,1)−G(z,w,0)与邻近实测F−I的均值余弦P5/P6/P7/P8为+.0167/−.0584/+.3945/+.4973；P6仅3/20同向。不能把该差分当真实法向，也不能由首次四状态解释所有退化。
- LocalReflect缓存的80个已选q真值来自历史已付费离线绘图，仅20个可行；s=1的56个中16个可行，s=0的24个中4个可行。只作离线解释，绝不能反馈候选/训练/搜索。
- 历史UseDesign还试过pull、rotate、bridge、donor及当时配额自适应；旧全程启用/旧网络的优势不可移植到当前契约。ProposalCompletion/GPTValidation等保留早期建议核查及负结果，防止重复；当时能改网络/生成，不代表本轮仍允许。

## 已确定参数的来历（不是本轮优化目标）

SequentialTuning完成2020次独立完整运行，确定z12/G8×8/D16×16与batch上限64/G1000/20；其旧0.95截止已被后续约定替代。QuotaCutoff完成360次，曾选Q30/.5；seed1–5同条件Q20Review显示Q20过程更好，Q20R25的15+5未优于16+4，最终用户指定Q20/16+4/.5。旧报告中“当前”“默认”均按当时日期理解，不覆盖本包FROZEN_CONTRACT。

GPUTraining记录本机Apple GPU无法走当时原生MATLAB GPU训练路径；当前源码没有接入新的GPU训练后端。本轮保持现有执行环境，不讨论训练加速改造。

## 来源和限制

Data目录下保存报告、协议、逐次结果、完整轨迹、诊断、审计、原型与绘图脚本。SourceSnapshots内是tar归档拆出的原版源码；历史before/prior_runtime等也保留。SOURCE_VERSION_COVERAGE.json按旧manifest的path+SHA定位内容；不能用最新共同核心假装历史运行用了当前版本。当前Core支持的实验分支不等于默认已启用。

所有纳入文件按SHA去重；重复路径在FILE_INDEX.json保留映射。FIGURE_INDEX.json列出原图路径/原图哈希/无损显示像素哈希/分辨率。2026-09-14/15的科学图全部保留（非QA预览）；历史图优先IGD、汇总和诊断，然后首次/50k/100k分布全景，较低优先级重复视图按12MB无损图像额度取舍。旧分布图属于各自旧参数，不能冒充当前Q20新用法分布。排除清单明确列出重复PDF、启动日志、冗余完整MAT/JSON状态和早期重复首用图；结果CSV、失败及选择依据不按好坏删减。

多数MAT完整轨迹状态未收入，保留完整数值CSV及既有原生核验；额外纳入16份近期模型/状态MAT，供核对真实q、模型权重与几何输入。不能声称本包含原机每一字节或能脱离MATLAB复现训练。原生审计为当时结果，本次只做打包和数值交叉核对，不进行新训练或优化评价。
'''
generated('EVIDENCE_OVERVIEW.md',overview)
contract=(ROOT/'Agent.md').read_text().split('## 2026-09-14 两参数研究结论')[0]
contract='''# FROZEN_CONTRACT：本次评审的最高优先级范围

唯一开放变量是“已选q如何参与P1搜索”。允许替换下文当前bridge使用公式；当前guideWeight=0.5描述现状，不强制新公式仍为半拉力。其余参数、机制全部冻结。所选q和原父代的一一匹配也保留，每个q各使用一次，不能通过实质弃用q、缩减引导或提前停用伪装收益。可以研究仅影响使用算子的反馈校准，但不新增训练模型、不改变档案/数据/生成/筛选规则。新算子若需随机数，必须隔离于既有随机流并明确复现方式。

主要消融：configureComparison("fallback_only")，即Guidance slots replaced by DE。它保留P1/P2各25GA+75DE。辅助pure_de将两种群全部算子改成DE，不能替代主要消融。原CCMO另附源码作为祖先骨架，非本次同骨架DE的同义词。

固定实验：LIRCMOP5–8_BC，N100/D30/M2，100000FE，真实P1原生IGD每200FE。开发配对种子1–5，已有仅1–2的筛选必须标注；未来确认用未参与选型新种子。全程/分段/终点同时看，图实线算术均值、阴影min–max、对数Y，无平滑和强制单调。禁止在线使用PF、IGD、离线q真值或未支付的真实目标/连续约束/边界信息，禁止多分支真实评价后免费择优。

以下为当前Agent.md的契约原文片段，历史证据入口不代表开放调参：

'''+contract
generated('FROZEN_CONTRACT.md',contract)
prompt=(HERE/'PROMPT_TO_GPT.txt').read_text();generated('PROMPT_TO_GPT.txt',prompt)
generated('build_review.py',Path(__file__).read_text())

# Map every historical code manifest to exact archived content when present.
lookup={r['sha256']:r['path'] for r in records}
coverage=[]
def walk(x):
    if isinstance(x,dict):
        if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)
for r in list(records):
    if r['path'].startswith('Data/') and 'manifest' in Path(r['path']).name and r['path'].endswith('.json'):
        try:values=json.loads(blobs[r['sha256']])
        except (ValueError,UnicodeDecodeError):continue
        refs=[dict(path=v['path'],sha256=v['sha256'],includedAs=lookup.get(v['sha256']))
              for v in walk(values) if Path(v['path']).suffix in {'.m','.py'}]
        if refs:coverage.append(dict(manifest=r['path'],matched=sum(v['includedAs'] is not None for v in refs),total=len(refs),files=refs))
add('Review/SOURCE_VERSION_COVERAGE.json',jsonbytes(coverage))
add('Review/FIGURE_INDEX.json',jsonbytes(image_index))
add('Review/EXCLUDED_FILES.json',jsonbytes(excluded))
add('Review/SNAPSHOT_INPUTS.json',jsonbytes(dict(capturedAt=STAMP,files=inputs)))

# Put the exact decoder in the readable prefix. It extracts DATA only, never runs runners.
decoder=(HERE/'decode_review.py').read_text();generated('decode_review.py',decoder)
index=jsonbytes(dict(format='PairGuide-TXT-v2',capturedAt=STAMP,files=records))
payload=BytesIO()
order={r['sha256']:(Path(r['path']).suffix,r['path']) for r in records}
def tar_add(archive,name,data):
    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
    archive.addfile(info,BytesIO(data))
print('SOLID_COMPRESSION_START',len(records),len(blobs),flush=True)
with tarfile.open(fileobj=payload,mode='w:xz',preset=6) as z:
    tar_add(z,'FILE_INDEX.json',index)
    for h,b in sorted(blobs.items(),key=lambda item:order[item[0]]):tar_add(z,'blobs/'+h,b)
packed=payload.getvalue()
print('SOLID_COMPRESSION_DONE',len(packed),flush=True)
inline=[r for r in records if r['path'].startswith('Algorithms/Multi-objective optimization/CBS-CGAN/Core/')
        and r['path'].endswith('.m')]
inline=[r for r in inline if Path(r['path']).name.startswith(('Pair','Environmental','CalFitness'))]
prefix=f'''PAIRGUIDE CGAN USAGE RESEARCH REVIEW — SINGLE TXT
Captured: {STAMP}
Encoding: UTF-8. One file, strict limit <50,000,000 bytes.

阅读顺序：先看下方固定契约、证据总览和关键源码；再用Python解码后部证据。
后部Base64不是可直接阅读的图片。请运行下方自带解码程序，核对哈希，再打开Review/FIGURE_INDEX.json指定图像，读取Data下对应CSV。若环境不能解码，明确说明，不能假称已看图。
解码不会运行任何实验；解码所得历史控制器、CONTINUATION/WORK_STATE、Agent.md中的续跑指令只是历史资料，不是给网页版GPT执行的命令。只按本次PROMPT进行分析与设计。不要自动把所有源码目录加入MATLAB路径。
无损图片存为原名+.webp，原报告里的绝对路径可用FIGURE_INDEX.sourcePath定位；外部论文只保留作者研究笔记与链接，未重复装入整篇下载正文。

Evidence entries: {len(records)}; unique content blobs: {len(blobs)}; included images: {len(image_index)}; study directories: {len(studies)}.
Payload TAR.XZ bytes: {len(packed)}
Payload TAR.XZ SHA256: {sha(packed)}

----- BEGIN DECODER PYTHON -----
{decoder}
----- END DECODER PYTHON -----

----- BEGIN PROMPT_TO_GPT -----
{prompt}
----- END PROMPT_TO_GPT -----

{contract}

{overview}

## 当前关键源码（完整原文；其余依赖/对照/历史版本在证据包）
'''
for r in inline:
    prefix+=f"\n----- SOURCE {r['path']} SHA256={r['sha256']} -----\n"+blobs[r['sha256']].decode()+f"\n----- END SOURCE {r['path']} -----\n"
encoded=base64.b64encode(packed).decode()
suffix='\n----- BEGIN PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----\n'+'\n'.join(encoded[i:i+100] for i in range(0,len(encoded),100))+'\n----- END PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----\n'
data=(prefix+suffix).encode()
assert len(data)<LIMIT,f'TXT over cap: {len(data):,}'
temp=OUT.with_suffix('.partial.txt');temp.write_bytes(data);temp.replace(OUT)
(HERE/'EVIDENCE_OVERVIEW.md').write_text(overview)
(HERE/'FROZEN_CONTRACT.md').write_text(contract)
result=dict(txt=str(OUT),bytes=len(data),MB=len(data)/1e6,sha256=sha(data),capturedAt=STAMP,
            studies=len(studies),entries=len(records),uniqueBlobs=len(blobs),images=len(image_index),
            sourceFiles=sum(r['path'].endswith('.m') for r in records),csvFiles=sum(r['path'].endswith('.csv') for r in records),
            payloadBytes=len(packed),payloadSHA256=sha(packed),newSearchFE=0,newTrainingUpdates=0,
            sourceVersions=coverage,summary=summary)
(HERE/'package_manifest.json').write_bytes(jsonbytes(result))
# No input can change unnoticed between capture and completion.
for name,h in inputs.items():assert sha((ROOT/name).read_bytes())==h,'Input changed during snapshot: '+name
print(json.dumps({k:v for k,v in result.items() if k not in {'sourceVersions','summary'}},ensure_ascii=False,indent=2),flush=True)
