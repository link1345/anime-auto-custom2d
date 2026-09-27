"""Reproduce a manual-calibrated Nagi rig demo; GPU depth is a separate job."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.project import Store
from src.pipeline import create_project
from src.character import manual_joint,build_rig,split_arm,fit_expression
from src.glb import export_glb


def build(output):
    output=Path(output);store=Store(output/'projects')
    p=create_project(store,(ROOT/'assets/nagi.png').read_bytes(),'Nagi / 手動関節デモ')
    for name,xy in json.loads((ROOT/'assets/nagi-rig-preset.json').read_text('utf-8'))['joints'].items():p=manual_joint(p,name,xy)
    p=build_rig(p)
    for side in ('Left','Right'):p=split_arm(store,p,'arm'+side,side)
    head=next(x for x in p['parts'] if x['id']=='head')
    p['expressions']=[{'id':'contour-demo','name':'輪郭モーフ動作確認','image':head['image'],'rect':[0,0,1,1],'meshVersion':p['meshVersion'],'target':'face'}]
    source=[[.2,.03],[.8,.03],[.8,.35],[.2,.35],[.433,.194],[.568,.194],[.47,.251],[.532,.251],[.5,.29]]
    target=[list(x) for x in source];target[6][1]-=.009;target[7][1]-=.009
    p=fit_expression(store,p,'contour-demo','head',source,target)
    p['warnings']=['デモ専用の手動関節。自動推定ではありません。','この再現スクリプトはGPU推論を実行しません。深度・See-throughは別ジョブから取り込んでください。']
    store.save(p)
    (output/'nagi-rig.glb').write_bytes(export_glb(store,p));(output/'nagi-rig.aac2d').write_bytes(store.export(p['id'],True))
    print(output/'nagi-rig.aac2d');print(output/'nagi-rig.glb')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/rig-demo')
    build(parser.parse_args().output)
