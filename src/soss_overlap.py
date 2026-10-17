"""Simultaneous native-column profile fitting and extraction for SOSS.

At each detector column D = sum_o F_o P_o + noise. Fluxes are independent
between orders/wavelengths. Shared intrinsic coefficients are fitted jointly,
profiling out the small linear flux system. This is a column-wise extension of
optimal extraction, not the wavelength/LSF-coupled ATOCA algorithm.
"""
from dataclasses import replace
import time

import numpy as np

from .shared_profile import _q, _geometry, _pixel_profile, _fit_coefficients


def unmodeled_pixels(orders, centers, shape, radius):
    """Do not interpret an unavailable order's pixels as uncontaminated sky."""
    mask=np.zeros(shape,bool)
    for order,trace in orders.items():
        missing=~trace['model_columns']
        x=trace['columns'][missing]
        if len(x):
            mask[:,x] |= np.abs(np.arange(shape[0])[:,None]-centers[order][missing])<=radius+1
    return mask


def solve_flux(profiles, data, variance, valid, condition_limit=1e6):
    """Weighted multi-order solve and full conditional covariance, per column.

    profiles is (column,row,order). Absent/unobserved orders get zero precision;
    ill-conditioned columns return no flux estimate rather than a pseudoinverse.
    The condition threshold uses a diagonally normalized normal matrix.
    """
    good = valid & np.isfinite(data) & (data != -9999) & np.isfinite(variance) & (variance > 0)
    good &= np.any(profiles > 0, axis=2)
    iv = np.divide(1., variance, out=np.zeros_like(variance, dtype=float), where=good)
    normal = np.einsum('jri,jrk,jr->jik', profiles, profiles, iv, optimize=True)
    rhs = np.einsum('jri,jr->ji', profiles, np.where(good, data, 0.)*iv, optimize=True)
    diagonal = np.diagonal(normal, axis1=1, axis2=2)
    observed = diagonal > 0
    scale = np.sqrt(np.where(observed, diagonal, 1.))
    correlation = normal/scale[:,:,None]/scale[:,None,:]
    o = np.arange(profiles.shape[2])
    correlation[:,o,o] = 1.
    eigenvalues = np.linalg.eigvalsh(correlation)
    condition = np.divide(eigenvalues[:,-1], eigenvalues[:,0],
                          out=np.full(len(profiles), np.inf), where=eigenvalues[:,0] > 0)
    supported = (condition < condition_limit) & np.any(observed, axis=1)
    covariance = np.zeros_like(normal)
    covariance[supported] = np.linalg.inv(correlation[supported])/scale[supported,:,None]/scale[supported,None,:]
    covariance *= observed[:,:,None] & observed[:,None,:]
    flux = np.einsum('jik,jk->ji', covariance, rhs)
    return flux, covariance, observed & supported[:,None], condition, iv


class _JointFit:
    def __init__(self, models, traces, columns, data, variance):
        self.models, self.traces, self.columns = models, traces, columns
        self.shape = np.shape(data[0])
        self.frames = []
        self.curvature_calls=0
        self.slices, offset = [], 0
        for model in models:
            self.slices.append(slice(offset, offset+model.coefficients.size))
            offset += model.coefficients.size
        self.size = offset
        cache={}
        for t, (image, var) in enumerate(zip(data, variance)):
            geometry = []
            for i,(model, centers) in enumerate(zip(models, traces)):
                key=(i,np.asarray(centers[t]).tobytes())
                if key not in cache:
                    rows, inside, count = _geometry(centers[t], self.shape[0], model.aperture_radius, model.spacing)
                    q = _q(centers[t], rows, inside, model.spacing, count)
                    cache[key]=(rows,inside,q)
                geometry.append(cache[key])
            self.frames.append((np.asarray(image).T, np.asarray(var).T, geometry))
        self.pairs=[]
        for i,xi in enumerate(columns):
            for j in range(i,len(models)):
                x,ii,jj=np.intersect1d(xi,columns[j],return_indices=True)
                if len(x):
                    self.pairs.append((i,j,x,ii,jj))

    def evaluate(self, values, unused_basis=None, curvature=False):
        value, gradient = 0., np.zeros(self.size)
        hessian = np.zeros((self.size,self.size)) if curvature else None
        self.reports=[[] for _ in self.models]
        profile_cache={}
        blocks={} if curvature else None
        for frame_index,(data,var,geometry) in enumerate(self.frames):
            signature=tuple(id(g[2]) for g in geometry)
            if signature not in profile_cache:
                profiles = np.zeros((*data.shape,len(self.models)))
                local = []
                for i,(model,x,(rows,inside,q),sl) in enumerate(zip(self.models,self.columns,geometry,self.slices)):
                    coefficients = values[sl].reshape(model.coefficients.shape)
                    z = model.basis@coefficients.T@model.contrast.T+model.mean
                    p,weighted = _pixel_profile(q,z)
                    profiles[x[:,None],rows,i] = p
                    jac = weighted@model.contrast-p[...,None]*(weighted.sum(axis=1)@model.contrast)[:,None,:]
                    local.append((p,jac))
                profile_cache[signature]=(profiles,local)
            profiles,local=profile_cache[signature]
            flux,cov,supported,_,iv = solve_flux(profiles,data,var,np.isfinite(data))
            # Missing columns are allowed, but a degenerate blend cannot train
            # distinguishable profiles from this likelihood.
            if np.any((np.sum(profiles**2*iv[:,:,None],axis=1)>0) & ~supported):
                raise ValueError('Degenerate overlapping training profiles')
            residual = np.einsum('jro,jo->jr',profiles,flux)-np.where(iv>0,data,0.)
            value += .5*np.sum(residual**2*iv)
            derivatives, cross = [], []
            for i,(model,x,(rows,inside,q),(p,jac),sl) in enumerate(zip(self.models,self.columns,geometry,local,self.slices)):
                derivative=jac*flux[x,i,None,None]
                weight=iv[x[:,None],rows]
                self.reports[i].append(dict(index=frame_index,
                    chi2=float(np.sum(residual[x[:,None],rows]**2*weight)),
                    valid_pixels=int(np.count_nonzero(weight)),
                    flux_min=float(np.min(flux[x,i])),flux_max=float(np.max(flux[x,i])),
                    flux_median=float(np.median(flux[x,i]))))
                grad=np.einsum('jrk,jr->jk',derivative,residual[x[:,None],rows]*weight)
                gradient[sl] += (grad.T@model.basis).ravel()
                derivatives.append(derivative)
                cross.append(np.einsum('jrk,jro,jr->jko',derivative,
                    profiles[x[:,None],rows],weight,optimize=True))
            if curvature:
                for i,j,x,ii,jj in self.pairs:
                    ri,rj=geometry[i][0][ii],geometry[j][0][jj]
                    index=ri-rj[:,:1]
                    inside=(index>=0)&(index<rj.shape[1])
                    aligned=derivatives[j][jj[:,None],np.clip(index,0,rj.shape[1]-1)]*inside[...,None]
                    block=np.einsum('jrk,jrl,jr->jkl',derivatives[i][ii],aligned,iv[x[:,None],ri],optimize=True)
                    # Variable projection: eliminate all order flux increments.
                    block-=np.einsum('jko,jop,jlp->jkl',cross[i][ii],cov[x],cross[j][jj],optimize=True)
                    if (i,j) not in blocks:
                        blocks[i,j]=block
                    else:
                        blocks[i,j]+=block
        if curvature:
            # Dispersion bases are shared across integrations. Sum spatial
            # curvature first, then contract once using a BLAS matrix product.
            # This is the same Hessian, without repeating the dense expansion.
            for i,j,x,ii,jj in self.pairs:
                mi,mj=self.models[i],self.models[j]
                block=blocks[i,j]
                bb=mi.basis[ii,:,None]*mj.basis[jj,None,:]
                dense=block.reshape(len(x),-1).T@bb.reshape(len(x),-1)
                dense=dense.reshape(block.shape[1],block.shape[2],mi.basis.shape[1],mj.basis.shape[1])
                dense=dense.transpose(0,2,1,3).reshape(mi.coefficients.size,mj.coefficients.size)
                si,sj=self.slices[i],self.slices[j]
                hessian[si,sj]=dense
                if i!=j:
                    hessian[sj,si]=dense.T
        if curvature:
            self.curvature_calls+=1
            if self.curvature_calls%25==0:
                print(f'  joint profile iteration {self.curvature_calls}, pixel chi2={2*value:.6g}',flush=True)
        return value,gradient,hessian,[]


def train_joint(models, traces, columns, data, variance, maxiter=500):
    """Joint variable-projection Gauss-Newton fit with coefficient N(0,I) priors."""
    started=time.perf_counter()
    executor=_JointFit(models,traces,columns,data,variance)
    timing={'coefficient_objective_seconds':0.}
    initial=np.concatenate([m.coefficients.ravel() for m in models])
    fitted=_fit_coefficients(executor,None,initial,maxiter,timing)
    if not fitted.success:
        raise RuntimeError('Joint SOSS profile fit did not converge: '+str(fitted.message))
    diagnostics=dict(algorithm='joint native-column variable projection',
        seconds=time.perf_counter()-started,iterations=int(fitted.nit),
        message=str(fitted.message),objective=float(fitted.fun),
        coefficient_count=len(initial),hyperparameters_optimized=False)
    output=[]
    for model,sl,report in zip(models,executor.slices,executor.reports):
        output.append(replace(model,coefficients=fitted.x[sl].reshape(model.coefficients.shape),
            covariance=None,diagnostics=dict(model.diagnostics,joint_fit=diagnostics,
                seed_training=model.diagnostics['training'],training=report)))
    return output,diagnostics


def extract(models, columns, centers, frame, variance, valid, aperture_radius, options):
    """Deblend all supplied orders, propagating the full flux covariance."""
    from .soss import aperture_weights
    profiles=np.zeros((frame.shape[1],frame.shape[0],len(models)))
    fractions=np.zeros((frame.shape[1],len(models)))
    for i,(model,x,y) in enumerate(zip(models,columns,centers)):
        p=model.evaluate(y,dispersion_coordinates=x)
        profiles[x,:,i]=p.T
        fractions[x,i]=np.sum(p*aperture_weights(len(frame),y,aperture_radius),axis=0)
    mask=valid.T.copy()
    for _ in range(frame.shape[0]+1):
        flux,cov,supported,condition,iv=solve_flux(profiles,frame.T,variance.T,mask,
            condition_limit=options['condition_limit'])
        if options['nsigma']==0:
            break
        residual=frame.T-np.einsum('jro,jo->jr',profiles,flux)
        score=np.where(iv>0,np.abs(residual)*np.sqrt(iv),-np.inf)
        worst=score.argmax(axis=1)
        reject=score[np.arange(len(score)),worst]>options['nsigma']
        if not np.any(reject):
            break
        mask[np.flatnonzero(reject),worst[reject]]=False
    observed=np.sum(profiles*(iv>0)[...,None],axis=1)
    supported &= observed >= options['minimum_profile_fraction']
    flux=np.where(supported,flux*fractions,np.nan)
    covariance=cov*fractions[:,:,None]*fractions[:,None,:]
    covariance=np.where(supported[:,:,None]&supported[:,None,:],covariance,np.nan)
    error=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2))
    return flux,error,observed,covariance,condition
