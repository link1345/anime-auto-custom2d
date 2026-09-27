"""Replace named expression records without overwriting image files needed by Undo."""
from .project import uid


def upsert_named(project,collection,record):
    name=str(record.get('name','')).strip()[:80].strip()
    if not name:raise ValueError('表情名を入力してください')
    items=project.setdefault(collection,[])
    matches=[item for item in items if str(item.get('name','')).strip()==name]
    ids={item['id'] for item in matches}
    record={**record,'id':matches[0]['id'] if matches else record.get('id',uid()),'name':name}
    if matches:
        index=items.index(matches[0]);items[:]=[item for item in items if item['id'] not in ids];items.insert(index,record)
        if collection=='expressionSources':
            for expression in project.get('expressions',[]):
                if expression.get('sourceId') in ids:
                    expression['sourceId']=record['id'];expression['stale']=True;expression['staleReason']='source-replaced'
                    expression.pop('driver',None)
                    for part in project.get('parts',[]):part.get('mesh',{}).get('morphs',{}).pop(expression['id'],None)
        else:
            for part in project.get('parts',[]):
                for eid in ids:part.get('mesh',{}).get('morphs',{}).pop(eid,None)
    else:items.append(record)
    return record
