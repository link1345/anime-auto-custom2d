"""glTF 2.0 binary exporter: embedded textures, skin/IBMs and expression targets."""
import json
import struct
import numpy as np


def export_glb(store,p):
    rig=p.get('rig')
    if not rig or rig.get('stale') or p.get('stale'):raise ValueError('有効なリグを構築してからGLBを出力してください')
    blob=bytearray();g={'asset':{'version':'2.0','generator':'Nagi Studio'},'scene':0,'scenes':[{'nodes':[0]}],
        'nodes':[{'name':p['name'],'children':[]}],'buffers':[],'bufferViews':[],'accessors':[],
        'meshes':[],'skins':[],'images':[],'textures':[],'materials':[],
        'extensionsUsed':['KHR_materials_unlit']}
    def view(raw,target=None):
        while len(blob)%4:blob.append(0)
        offset=len(blob);blob.extend(raw);v={'buffer':0,'byteOffset':offset,'byteLength':len(raw)}
        if target:v['target']=target
        g['bufferViews'].append(v);return len(g['bufferViews'])-1
    def accessor(data,dtype,kind,component,target=None,bounds=False):
        a=np.asarray(data,dtype=dtype);v=view(a.tobytes(),target)
        item={'bufferView':v,'componentType':component,'count':len(a),'type':kind}
        if bounds:item.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
        g['accessors'].append(item);return len(g['accessors'])-1
    aspect=p['size'][0]/p['size'][1]
    def position(x):return [(x[0]-.5)*aspect,1-x[1],x[2]]
    bones=rig['bones'];ibms=[]
    for i,bone in enumerate(bones):
        point=np.asarray(position(bone['position']));parent=bone['parent']
        local=point-np.asarray(position(bones[parent]['position'])) if parent>=0 else point
        node={'name':bone['name'],'translation':local.tolist(),'children':[]};g['nodes'].append(node)
        g['nodes'][parent+1 if parent>=0 else 0]['children'].append(i+1)
        matrix=np.eye(4,dtype='float32');matrix[:3,3]=-point;ibms.append(matrix.T.reshape(16))
    g['skins'].append({'name':'CharacterRig','joints':list(range(1,len(bones)+1)),'skeleton':1,
        'inverseBindMatrices':accessor(ibms,'<f4','MAT4',5126)})
    expression_ids=[e['id'] for e in p['expressions'] if not e.get('stale') and e.get('fit')]
    for part in p['parts']:
        if not part.get('visible',True) or (p.get('display',{}).get('templatesVisible') is False and part.get('provenance')=='template'):continue
        mesh=part['mesh']
        if 'weights' not in mesh:raise ValueError('スキン未生成のパーツがあります。リグを再構築してください')
        vertices=np.asarray(mesh['vertices']);positions=np.array([position(v) for v in vertices],dtype='float32')
        r=part.get('rect',[0,0,1,1]);uv=(vertices[:,:2]-r[:2])/r[2:]
        attrs={'POSITION':accessor(positions,'<f4','VEC3',5126,34962,True),
          'TEXCOORD_0':accessor(uv,'<f4','VEC2',5126,34962),
          'JOINTS_0':accessor(mesh['joints'],'<u2','VEC4',5123,34962),
          'WEIGHTS_0':accessor(mesh['weights'],'<f4','VEC4',5126,34962)}
        image_index=len(g['images']);raw=(store.folder(p['id'])/part['image']).read_bytes()
        g['images'].append({'bufferView':view(raw),'mimeType':'image/png','name':part['name']})
        g['textures'].append({'source':image_index});mi=len(g['materials'])
        g['materials'].append({'name':part['name'],'doubleSided':True,'alphaMode':'BLEND',
            'pbrMetallicRoughness':{'baseColorTexture':{'index':image_index},'metallicFactor':0,'roughnessFactor':1},'extensions':{'KHR_materials_unlit':{}}})
        # Y is flipped, so reverse winding.
        indices=np.asarray(mesh['triangles'])[:,[0,2,1]].reshape(-1)
        primitive={'attributes':attrs,'indices':accessor(indices,'<u2','SCALAR',5123,34963),'material':mi}
        targets=[]
        for eid in expression_ids:
            delta=np.asarray(mesh.get('morphs',{}).get(eid,np.zeros_like(vertices)))*[aspect,-1,1]
            targets.append({'POSITION':accessor(delta,'<f4','VEC3',5126,34962,True)})
        m={'name':part['name'],'primitives':[primitive]}
        if targets:primitive['targets']=targets;m.update(weights=[0]*len(targets),extras={'targetNames':expression_ids})
        mesh_index=len(g['meshes']);g['meshes'].append(m);index=len(g['nodes'])
        g['nodes'].append({'name':part['name'],'mesh':mesh_index,'skin':0});g['scenes'][0]['nodes'].append(index)
    g['extras']={'coordinateSystem':'normalized 2.5D portrait, height 1','limitations':['No volume behind the illustration','Hair collision simulation and expression texture blending are runtime features; not encoded by glTF'],
                 'expressionNames':{e['id']:e['name'] for e in p['expressions']}}
    for node in g['nodes']:
        if node.get('children')==[]:node.pop('children')
    while len(blob)%4:blob.append(0)
    g['buffers']=[{'byteLength':len(blob)}]
    data=json.dumps(g,ensure_ascii=False,separators=(',',':')).encode();data+=b' '*((-len(data))%4)
    length=12+8+len(data)+8+len(blob)
    return struct.pack('<4sII',b'glTF',2,length)+struct.pack('<I4s',len(data),b'JSON')+data+struct.pack('<I4s',len(blob),b'BIN\0')+blob
