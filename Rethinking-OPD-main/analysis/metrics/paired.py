"""Same-checkpoint student/privileged-teacher geometry on aligned fixed inputs."""
import numpy as np

from .neighbors import knn, neighbor_overlap
from .rsa import align_rows, compare_rsms, cosine_rsm
from .spectrum import linear_cka


def paired_geometry(student,teacher,student_ids,teacher_ids,ks=(8,16,32),max_pairs=None,seed=0,cka=True):
    """Use one jointly finite/nonzero cohort for all direct comparison metrics.

    The caller supplies the SAME fixed response IDs, layer and response-relative
    anchor. Token positions in the teacher prompt may be shifted by reference text.
    Neighborhood indices refer to valid_sample_ids, never the unfiltered cohort.
    """
    student=np.asarray(student,dtype=np.float64)
    teacher=np.asarray(teacher,dtype=np.float64)
    if student.ndim!=2 or teacher.ndim!=2 or student.shape!=teacher.shape:
        raise ValueError('Paired geometry requires matching [questions, hidden_size] arrays')
    if len(student_ids)!=len(student) or len(teacher_ids)!=len(teacher):
        raise ValueError('One unique sample ID per representation row required')
    teacher=align_rows(teacher,teacher_ids,student_ids)
    # align_rows also validates unique student IDs through exact set equality.
    valid=(np.isfinite(student).all(1)&np.isfinite(teacher).all(1)
           &(np.linalg.norm(student,axis=1)>0)&(np.linalg.norm(teacher,axis=1)>0))
    ids=np.asarray(student_ids)
    invalid_ids=ids[~valid].tolist()
    ids=ids[valid].tolist()
    h,t=student[valid],teacher[valid]
    r,_=cosine_rsm(h)
    rt,_=cosine_rsm(t)
    rsa=compare_rsms(r,rt,max_pairs,seed)
    metrics={'rsa_rho_student_teacher':rsa['rho'], 'rsa_rmse_student_teacher':rsa['rmse']}
    if cka:
        metrics['linear_cka_student_teacher']=linear_cka(h,t) if len(h)>1 else np.nan
    neighborhoods={}
    for k in ks:
        if not isinstance(k,int) or isinstance(k,bool) or k<1:
            raise ValueError('Paired geometry k must be a positive integer')
        name=f'neighbor_overlap_student_teacher_k{k}'
        if len(ids)<=k:
            metrics[name]=np.nan
            continue
        sindex,sdist=knn(h,ids,k)
        tindex,tdist=knn(t,ids,k)
        overlap=neighbor_overlap(sindex,tindex)
        metrics[name]=float(overlap.mean())
        neighborhoods[k]=dict(student_indices=sindex,teacher_indices=tindex,
                              student_distances=sdist,teacher_distances=tdist,per_sample_overlap=overlap)
    return dict(metrics=metrics,n_valid=len(ids),n_pairs=rsa['n_pairs'],
                valid_sample_ids=ids,invalid_sample_ids=invalid_ids,neighborhoods=neighborhoods)


def summarize_student_teacher(arrays,metadata,monitor,emit,cohorts=None):
    """Join by pool/split/layer/anchor and emit a separate comparison role."""
    lookup={}
    for key,meta in metadata.items():
        identity=tuple(meta[k] for k in ('role','pool','split','layer','anchor'))
        if identity in lookup:
            raise ValueError('Duplicate geometry cohort metadata')
        lookup[identity]=key
    if cohorts is None:
        cohorts=[identity[1:] for identity in lookup if identity[0]=='student']
    manifests,neighbors=[],{}
    for cohort in cohorts:
        pool,split,layer,anchor=cohort
        if pool=='live_rollout':
            continue
        key=lookup.get(('student',*cohort))
        teacher_key=lookup.get(('privileged_teacher',*cohort))
        if (key is None)!=(teacher_key is None):
            raise ValueError('Missing same-checkpoint teacher geometry for a fixed student cohort')
        h=arrays[key] if key is not None else np.empty((0,0))
        t=arrays[teacher_key] if teacher_key is not None else np.empty((0,0))
        ids=metadata[key]['sample_ids'] if key is not None else []
        tids=metadata[teacher_key]['sample_ids'] if teacher_key is not None else []
        result=paired_geometry(h,t,ids,tids,
                               monitor['knn_k'],monitor['rsa_max_pairs'],monitor['seed'],monitor.get('linear_cka',True))
        manifest=dict(pool=pool,split=split,layer=layer,anchor=anchor,
                      student_key=key,teacher_key=teacher_key,n_input=len(ids),
                      n_valid=result['n_valid'],n_pairs=result['n_pairs'],
                      valid_sample_ids=result['valid_sample_ids'],invalid_sample_ids=result['invalid_sample_ids'])
        manifests.append(manifest)
        for name,value in result['metrics'].items():
            is_neighbor=name.startswith('neighbor_overlap')
            extra=dict(comparison='same_checkpoint_student_vs_privileged_teacher',n_pairs=result['n_pairs'],
                       rsa_pair_seed=monitor['seed'],cohort_n=len(ids),
                       invalid_sample_ids=result['invalid_sample_ids'])
            if is_neighbor:
                k=int(name.rsplit('_k',1)[1])
                extra['chance_level']=k/(result['n_valid']-1) if result['n_valid']>k else None
            if not np.isfinite(value):
                extra['reason']='insufficient_or_degenerate_paired_geometry'
            emit(name,value,result['n_valid'],pool,split,layer,anchor,
                 role='student_vs_privileged_teacher',**extra)
        for k,values in result['neighborhoods'].items():
            prefix=f"student_teacher__{pool}__{split}__{anchor}__layer{layer}__k{k}"
            neighbors[prefix+'__sample_ids']=np.asarray(result['valid_sample_ids'])
            for name,value in values.items():
                neighbors[prefix+'__'+name]=value
    return manifests,neighbors
