"""Repackage the already verified TXT payload, without changing research evidence."""
from pathlib import Path
from io import BytesIO
import base64, hashlib, json, zipfile

HERE=Path(__file__).resolve().parent
txt=HERE/'PairGuide_CGAN_Usage_Review_20260915.txt'
raw=txt.read_text()
marker='----- BEGIN PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----'
prefix,encoded=raw.rsplit(marker,1)
encoded=encoded.split('----- END PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----',1)[0]
payload=base64.b64decode(''.join(encoded.split()),validate=True)
manifest=json.loads((HERE/'package_manifest.json').read_text())
assert hashlib.sha256(payload).hexdigest()==manifest['payloadSHA256']
out=HERE/'PairGuide_CGAN_Usage_WEB_20260915.zip'

web_note='''网页版备用包：与49.10MB完整TXT完全相同的证据。

请先阅读START_HERE.txt的固定契约、证据总览和当前关键源码，再读取PROMPT_TO_GPT.txt。
本ZIP内evidence.tar.xz就是完整TXT中的同一压缩证据，SHA-256完全一致，没有删减算法、结果或图片。
若可使用Python，运行：
    python restore_web_bundle.py /path/to/PairGuide_CGAN_Usage_WEB_20260915.zip /tmp/pairguide_review
只会解包并核验数据，不会运行MATLAB、训练或实验控制器。
从解包后的Review/FIGURE_INDEX.json选择图片，直接显示对应.webp；Data下为CSV和报告；FILE_INDEX.json有全部文件及哈希。
先读当前用法研究，再按需读旧参数历史，避免将17,814条记录一次性塞进语言模型上下文。

官方文件上传说明：文本/文档每文件限制2M tokens，另有字节限制；所以大TXT即使不足50MB也可能被拒收。
https://help.openai.com/en/articles/8555545-file-uploads-faq
本地解包完整性已核验；未在用户网页版账号实际上传，ZIP接收及Python可用性仍取决于所用模式/账号。
'''
prompt=(HERE/'PROMPT_TO_GPT.txt').read_text()
prompt=prompt.replace('附件是单个TXT，后部有Base64编码的证据压缩包，包含实际图片、CSV、源码历史版本和诊断状态。请使用Python按内置解码程序读取，核对SHA-256，打开FIGURE_INDEX中的关键图片并读取对应CSV。',
'''若附件为完整TXT，按其中的Base64解码程序读取证据；若附件为ZIP，先读START_HERE.txt，再运行restore_web_bundle.py解包evidence.tar.xz。两者包含完全相同的图片、CSV、源码历史版本和诊断状态。请使用Python核对SHA-256，打开FIGURE_INDEX中的关键图片并读取对应CSV。''')
(HERE/'PROMPT_TO_GPT_WEB.txt').write_text(prompt)
decoder=(HERE/'decode_review.py').read_text()
start=decoder.index("    raw=Path(source).read_text")
end=decoder.index('    root=Path(destination)',start)
decoder=decoder[:start]+'''    with zipfile.ZipFile(source) as outer:
        assert outer.testzip() is None
        data=outer.read('evidence.tar.xz')
        expected=json.loads(outer.read('BUNDLE.json'))['payloadSHA256']
    assert hashlib.sha256(data).hexdigest()==expected,'Payload hash mismatch'
'''+decoder[end:]
decoder=decoder.replace('import base64, hashlib, json, re, sys, tarfile','import base64, hashlib, json, re, sys, tarfile, zipfile')
decoder=decoder.replace('decode_review.py /path/to/PairGuide_CGAN_Usage_Review_20260915.txt',
                        'restore_web_bundle.py /path/to/PairGuide_CGAN_Usage_WEB_20260915.zip')
(HERE/'restore_web_bundle.py').write_text(decoder)
(HERE/'START_HERE_WEB.txt').write_text(web_note+'\n\n'+prefix)
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.writestr('evidence.tar.xz',payload,compress_type=zipfile.ZIP_STORED)
    z.writestr('START_HERE.txt',(web_note+'\n\n'+prefix).encode())
    z.writestr('PROMPT_TO_GPT.txt',prompt.encode())
    z.writestr('restore_web_bundle.py',decoder.encode())
    z.writestr('BUNDLE.json',json.dumps({k:manifest[k] for k in ['capturedAt','entries','images','payloadSHA256']}).encode())
assert out.stat().st_size<50_000_000
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert z.read('evidence.tar.xz')==payload
result={'zip':str(out),'bytes':out.stat().st_size,'MB':out.stat().st_size/1e6,
        'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'samePayloadAsTXT':True}
(HERE/'web_package_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
