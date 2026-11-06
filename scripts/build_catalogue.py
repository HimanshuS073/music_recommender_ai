#!/usr/bin/env python3
"""Run from any directory: python scripts/build_catalogue.py [--workers 4].
Uses Python standard library and ffmpeg/ffprobe; raw data is read only.
"""
import math, argparse, ast, collections, concurrent.futures, csv, hashlib, json, os, pathlib, re, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/processed'
MAPPING = {'title': ('track','title'), 'artist_id': ('artist','id'), 'artist_name': ('artist','name'), 'album_id': ('album','id'), 'album_title': ('album','title'), 'genre_top': ('track','genre_top'), 'genres': ('track','genres'), 'subset': ('set','subset'), 'split': ('set','split')}
OPTIONAL_MAPPING = {'language_code': ('track', 'language_code'),
                    'album_date_released': ('album', 'date_released'),
                    'license': ('track', 'license')}
MAPPING.update(OPTIONAL_MAPPING)
OPTIONAL_FIELDS = (*OPTIONAL_MAPPING, 'license_missing')

def optional_metadata(source):
    values = {key: source[column] for key, column in OPTIONAL_MAPPING.items()}
    values['license_missing'] = str(not values['license'].strip())
    return values

def index_rows(rows):
    indexed = {}
    for row in rows:
        value = str(row['track_id'])
        if not re.fullmatch(r'[0-9]+', value):
            raise ValueError(f'Invalid track ID: {value!r}')
        tid = int(value)
        if tid in indexed:
            raise ValueError(f'Duplicate track ID: {tid}')
        indexed[tid] = row
    return indexed

def update_metadata():
    catalogue = OUT / 'catalogue.csv'
    with catalogue.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = [field for field in reader.fieldnames if field != 'release_year']
        original = list(reader)
    before = index_rows(original)
    if len(before) != 8000:
        raise ValueError('Expected exactly 8,000 catalogue rows')
    with (RAW / 'fma_metadata/tracks.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        columns = list(zip(next(reader), next(reader)))
        if not set(MAPPING.values()) <= set(columns):
            raise ValueError('Required source columns unavailable')
        if next(reader)[0] != 'track_id':
            raise ValueError('Missing original track ID index label')
        source = index_rows([dict(zip(columns, row)) | {'track_id': row[0]} for row in reader])
    unmatched = sorted(set(before) - set(source))
    if unmatched:
        raise ValueError(f'Unmatched catalogue IDs: {unmatched}')
    with (OUT / 'catalogue_audit.csv').open(newline='', encoding='utf-8') as f:
        audit = index_rows(list(csv.DictReader(f)))
    if set(audit) != set(before):
        raise ValueError('Existing audit ID set differs from catalogue')
    report = json.loads((OUT / 'catalogue_validation.json').read_text())
    for field in OPTIONAL_FIELDS:
        if field not in fields:
            fields.append(field)
    updated = [{key: value for key, value in dict(row, **optional_metadata(source[int(row['track_id'])])).items() if key != 'release_year'} for row in original]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=OUT, delete=False) as f:
            temporary = pathlib.Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(updated)
        with temporary.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise ValueError('Column readback mismatch')
            written = list(reader)
        after = index_rows(written)
        if len(written) != 8000 or set(after) != set(before):
            raise ValueError('Track ID set or row count changed')
        if 'release_year' in fields:
            raise ValueError('Removed column remains')
        missing_required = []
        missing_paths = []
        for tid, row in after.items():
            for key, value in before[tid].items():
                if key != 'release_year' and row[key] != value:
                    raise ValueError(f'Unrelated value changed: {tid}, {key}')
            if any(row[key] != value for key, value in optional_metadata(source[tid]).items()):
                raise ValueError(f'Metadata readback mismatch: {tid}')
            for key, column in MAPPING.items():
                if row[key] != source[tid][column]:
                    raise ValueError(f'Dataset metadata mismatch: {tid}, {key}')
            for key in ('track_id','title','artist_id','artist_name','genre_top','subset','split','audio_path'):
                if not row[key].strip(): missing_required.append((tid, key))
            path = ROOT / row['audio_path']
            if not path.is_file() or path.stat().st_size == 0:
                missing_paths.append(tid)
        if missing_paths:
            raise ValueError(f'Missing/empty paths: {missing_paths}')
        if missing_required:
            raise ValueError(f'Missing required metadata: {missing_required}')
        result = {'readback_validated': True, 'rows': len(written), 'unique_ids': len(after),
                  'id_set_unchanged': True, 'unmatched_ids': unmatched, 'missing_path_count': 0,
                  'unrelated_values_unchanged': True, 'audio_audit_reused': True,
                  'columns': fields, 'release_year_absent': True,
                  'missing_required_metadata_count': 0, 'all_retained_values_unchanged': True,
                  'optional_source_values_preserved': True,
                  'availability': {key: {'available': sum(bool(row[key].strip()) for row in written),
                                         'missing': sum(not row[key].strip() for row in written)} for key in OPTIONAL_FIELDS},
                  'license_missing_true': sum(row['license_missing'] == 'True' for row in written)}
        report['column_mapping'].update(OPTIONAL_MAPPING)
        report['column_mapping'].pop('release_year', None)
        report['column_mapping'].update({'license_missing': 'album-independent track.license null/empty/whitespace check'})
        report['column_provenance'] = {'dataset_metadata': ['track_id', *MAPPING], 'audio_properties': ['file_size_bytes','duration_seconds','sample_rate_hz','channels'], 'validation_flags': ['probe_ok','strict_decode_ok','audio_issue','title_tag_status','artist_tag_status','license_missing'], 'technical_locator': ['audio_path']}
        report['initial_features'] = {'numerical_similarity': 'MFCC statistics (not extracted in this task)', 'genre': 'evaluate separately', 'excluded': ['language_code','album_date_released','title','artist_id','license','license_missing']}
        report['metadata_update'] = result
        os.replace(temporary, catalogue)
        temporary = None
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=OUT, delete=False) as f:
            json.dump(report, f, indent=2); f.write('\n'); report_temp = f.name
        os.replace(report_temp, OUT / 'catalogue_validation.json')
        print(json.dumps(result, indent=2))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0

def run(args):
    return subprocess.run(args, capture_output=True, text=True, errors='replace')
def norm(s): return ' '.join(s.split()).casefold()
def atomic_csv(path, rows):
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=OUT, delete=False) as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); name=f.name
    os.replace(name,path)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=4); p.add_argument('--metadata-only', action='store_true'); args=p.parse_args()
    if args.metadata_only: return update_metadata()
    if args.workers<1 or args.workers>32: p.error('workers must be 1..32')
    OUT.mkdir(parents=True,exist_ok=True)
    tools={t:run([t,'-version']).stdout.splitlines()[0] for t in ('ffmpeg','ffprobe')}
    checks={}; checksum_results=[]
    for directory in (RAW/'fma_small', RAW/'fma_metadata'):
        manifest=directory/'checksums'
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                m=re.fullmatch(r'([0-9a-fA-F]{40})\s+\*?(.+)',line)
                if not m: checksum_results.append({'manifest':str(manifest.relative_to(ROOT)), 'unsupported_line':line}); continue
                path=directory/m[2]; checks[str(path.relative_to(ROOT))]=m[1].lower()
                if directory.name=='fma_metadata':
                    status='missing' if not path.is_file() else ('match' if hashlib.sha1(path.read_bytes()).hexdigest()==m[1].lower() else 'mismatch')
                    checksum_results.append({'path':str(path.relative_to(ROOT)),'status':status})
    with (RAW/'fma_metadata/tracks.csv').open(newline='') as f:
        r=csv.reader(f); cols=list(zip(next(r),next(r))); next(r)
        source=[dict(zip(cols,row))|{'track_id':row[0]} for row in r]
    rows=[]
    for s in source:
        if s['set','subset']!='small': continue
        tid=int(s['track_id']); code=f'{tid:06d}'
        rows.append({'track_id':tid, **{k:s[v] for k,v in MAPPING.items()},'audio_path':f'data/raw/fma_small/{code[:3]}/{code}.mp3'})
    for row in rows:
        row['license_missing']=str(not row['license'].strip())
    rows.sort(key=lambda r:r['track_id'])
    with (RAW/'fma_metadata/genres.csv').open() as f: genres={int(r['genre_id']) for r in csv.DictReader(f)}
    artists=collections.defaultdict(set)
    for r in rows: artists[r['artist_id']].add(r['artist_name'])
    required=['track_id','title','artist_id','artist_name','genre_top','subset','split','audio_path']
    def inspect(r):
        r=dict(r); path=ROOT/r['audio_path']; issues=[]
        r['missing_required']=json.dumps([k for k in required if not str(r[k]).strip()])
        r['metadata_link']='track_id'; r['artist_attribution']='not independently verified'
        r['regular_file']=path.is_file(); r['file_size_bytes']=path.stat().st_size if path.is_file() else 0
        r['filename_matches_id']=path.stem==f"{r['track_id']:06d}"
        r['duration_seconds']='unavailable'; r['sample_rate_hz']='unavailable'; r['channels']='unavailable'; r['sha256']=''; r['checksum_status']='not covered'; r['decoder_failure']=False; r['decoder_messages']=''; r['probe_error']=''
        tags={}
        r['probe_ok']=False; r['strict_decode_ok']=False
        r['intended_loader_status']='pending: loader not selected'
        if r['regular_file'] and r['file_size_bytes']:
            data=path.read_bytes(); r['sha256']=hashlib.sha256(data).hexdigest()
            if r['audio_path'] in checks: r['checksum_status']='match' if hashlib.sha1(data).hexdigest()==checks[r['audio_path']] else 'mismatch'
            probe=run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)])
            try:
                if probe.returncode: raise ValueError('ffprobe failed')
                info=json.loads(probe.stdout); streams=[s for s in info.get('streams',[]) if s.get('codec_type')=='audio']; stream=streams[0]
                r['duration_seconds']=stream.get('duration',info.get('format',{}).get('duration','')); r['sample_rate_hz']=stream.get('sample_rate',''); r['channels']=stream.get('channels','')
                for key in ('duration_seconds','sample_rate_hz','channels'):
                    try:
                        if not math.isfinite(float(r[key])) or float(r[key])<=0: raise ValueError('invalid property')
                    except (TypeError,ValueError): r[key]='unavailable'
                r['probe_ok']=all(r[k]!='unavailable' for k in ('duration_seconds','sample_rate_hz','channels'))
                if not r['probe_ok']: r['probe_error']='Missing or invalid measured audio properties'
                tags={k.casefold():v for section in [info.get('format',{}),stream] for k,v in section.get('tags',{}).items()}
            except (ValueError,IndexError): r['probe_error']=probe.stderr or 'No readable audio stream'
            decode=run(['ffmpeg','-nostdin','-v','warning','-xerror','-err_detect','explode','-i',str(path),'-map','0:a:0','-f','null','-'])
            r['strict_decode_ok']=decode.returncode==0
            r['decoder_failure']=decode.returncode!=0; r['decoder_messages']=decode.stderr
        else: r['checksum_status']='missing' if r['audio_path'] in checks else 'not covered'
        audio_issues=[]
        if not r['regular_file'] or not r['file_size_bytes']: audio_issues.append('missing_or_empty_audio')
        if not r['probe_ok']: audio_issues.append('probing_failure')
        if r['decoder_failure']: audio_issues.append('strict_decoder_rejection')
        if r['decoder_messages']: audio_issues.append('decoder_messages')
        r['audio_issue']='; '.join(audio_issues) or 'none'
        for tag,col in [('title','title'),('artist','artist_name')]:
            r['embedded_'+tag]=tags.get(tag,''); r[tag+'_tag_status']='unavailable' if not norm(tags.get(tag,'')) else ('agree' if norm(tags[tag])==norm(r[col]) else 'disagree')
        try: r['genre_ids_resolve']=all(isinstance(g,int) and g in genres for g in ast.literal_eval(r['genres']))
        except (ValueError,SyntaxError,TypeError): r['genre_ids_resolve']=False
        r['split_valid']=r['split'] in {'training','validation','test'}
        r['artist_name_consistent']=len(artists[r['artist_id']])==1
        r['audio_integrity_verified']=bool(r['regular_file'] and r['file_size_bytes'] and not r['decoder_failure'] and not r['probe_error'] and r['checksum_status']!='mismatch')
        for k in ('genre_ids_resolve','split_valid','artist_name_consistent','filename_matches_id','audio_integrity_verified'):
            if not r[k]: issues.append(k+' failed')
        if json.loads(r['missing_required']): issues.append('missing required metadata')
        if r['decoder_messages']: issues.append('decoder messages; review')
        for tag in ('title','artist'):
            if r[tag+'_tag_status']!='agree': issues.append(tag+' tag '+r[tag+'_tag_status'])
        r['issues']=json.dumps(issues); return r
    audits=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(inspect,rows),1):
            audits.append(r)
            if i%200==0: print(f'Audited {i}/{len(rows)}',flush=True)
    hashes=collections.defaultdict(list)
    for r in audits:
        if r['sha256']: hashes[r['sha256']].append(r['track_id'])
    duplicates=[v for v in hashes.values() if len(v)>1]
    for r in audits: r['duplicate_audio_track_ids']=json.dumps(hashes[r['sha256']] if r['sha256'] and len(hashes[r['sha256']])>1 else [])
    affected={name:[r['track_id'] for r in audits if pred(r)] for name,pred in {
        'missing_or_empty_audio':lambda r:not r['regular_file'] or not r['file_size_bytes'], 'decoding_failures':lambda r:r['decoder_failure'], 'missing_required_metadata':lambda r:bool(json.loads(r['missing_required'])), 'checksum_mismatches':lambda r:r['checksum_status']=='mismatch', 'probe_failures':lambda r:bool(r['probe_error']), 'decoder_warnings':lambda r:bool(r['decoder_messages']), 'genre_issues':lambda r:not r['genre_ids_resolve'], 'split_issues':lambda r:not r['split_valid'], 'artist_inconsistencies':lambda r:not r['artist_name_consistent']}.items()}
    passed=len(rows)==8000 and len({r['track_id'] for r in rows})==8000 and not any(affected[k] for k in ('missing_or_empty_audio','missing_required_metadata'))
    report={'passed':passed,'rows':len(rows),'unique_track_ids':len({r['track_id'] for r in rows}),'counts':{k:len(v) for k,v in affected.items()},'affected_track_ids':affected,'missing_required_field_counts':{k:sum(k in json.loads(r['missing_required']) for r in audits) for k in required},'tag_comparison':{t:dict(collections.Counter(r[t+'_tag_status'] for r in audits)) for t in ('title','artist')},'checksum_coverage':dict(collections.Counter(r['checksum_status'] for r in audits)),'metadata_checksums':checksum_results,'duplicate_audio_groups':duplicates,'unexpected_mp3_files':sorted(str(p.relative_to(ROOT)) for p in (RAW/'fma_small').rglob('*') if p.suffix.lower()=='.mp3' and str(p.relative_to(ROOT)) not in {r['audio_path'] for r in rows}),'tools':tools,'column_mapping':MAPPING,'artist_attribution':'not independently verified','duration_source':'ffprobe audio stream duration, falling back to format duration','split_values':['training','validation','test']}
    report['catalogue_complete']=passed
    report['audio_readiness']={'probe_ok':sum(r['probe_ok'] for r in audits), 'strict_decode_ok':sum(r['strict_decode_ok'] for r in audits), 'intended_loader':{'status':'pending', 'reason':'No loader selected in project source, dependencies, or development plan; Stage 2 selects loading configuration.', 'tested':0, 'passed':None, 'failed':None, 'empty_output_check':'pending', 'nonfinite_samples_check':'pending', 'invalid_duration_check':'pending'}}
    report['six_track_investigation']=[{k:r[k] for k in ('track_id','probe_ok','strict_decode_ok','probe_error','decoder_messages','intended_loader_status')} for r in audits if r['track_id'] in {98565,98567,98569,99134,108925,133297}]
    atomic_csv(OUT/'catalogue_audit.csv',audits)
    (OUT/'catalogue_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if passed:
        fields=list(rows[0])+['file_size_bytes','duration_seconds','sample_rate_hz','channels','probe_ok','strict_decode_ok','audio_issue','title_tag_status','artist_tag_status']; atomic_csv(OUT/'catalogue.csv',[{k:r[k] for k in fields} for r in audits])
        with (OUT/'catalogue.csv').open(newline='') as f: written=list(csv.DictReader(f))
        ids=[int(r['track_id']) for r in written]
        assert len(written)==len(set(ids))==8000 and ids==sorted(ids)
        assert [(r['track_id'],r['audio_path']) for r in written]==[(str(r['track_id']),r['audio_path']) for r in rows]
        assert all((ROOT/r['audio_path']).is_file() and (ROOT/r['audio_path']).stat().st_size>0 for r in written)
        report['catalogue_readback']={'verified':True,'rows':len(written),'unique_ids':len(set(ids)),'audio_paths_verified':len(written)}
    (OUT/'catalogue_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','rows','unique_track_ids','counts','tag_comparison','checksum_coverage')},indent=2))
    if passed: return update_metadata()
    return 1
if __name__=='__main__': raise SystemExit(main())
