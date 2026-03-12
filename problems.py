import dolfinx as df
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc

class Problem:
    """A base problem class"""
    def __init__(self, **kwargs):
        self.update(**kwargs)
    
    def __del__(self):
        if hasattr(self, '_vtxs'): 
            for vtx in self._vtxs.values(): 
                if vtx is not None: vtx.close()
    
    @property
    def allowed_input_parameters(self):
        return ('output_basepath', 'comm', 'vis_e')

    def update(self, **kwargs):
        self.reset()

        self.output_basepath = None
        self.comm = MPI.COMM_WORLD
        self.vis_e = ('Lagrange', 1)

        for k, v in kwargs.items():
            if k in self.allowed_input_parameters and hasattr(self, k):
                setattr(self, k, v)

    def reset(self):
        for attr in [
                    '_mesh', '_Vs', '_x',
                    '_u_i', '_u_a', '_u_t', '_bcs', '_ns', 
                    '_F', '_JF', '_Fvec', '_JFmat', 
                    '_ds', '_dS', '_dx', 
                    '_u_vis', '_vtxs'
                    ]:
            if hasattr(self, attr): delattr(self, attr)

    @property
    def meshdata(self):
        if not hasattr(self, '_meshdata'):
            raise NotImplementedError("meshdata not implemented")
        return self._meshdata
    
    @property
    def entity_maps(self):
        if not hasattr(self, '_entity_maps') or self._entity_maps is None:
            _ = self.meshdata
            if not hasattr(self, '_entity_maps'): self._entity_maps = None
        return self._entity_maps

    @meshdata.setter
    def meshdata(self, i):
        self._meshdata = i
    
    @property
    def mesh(self):
        return self.meshdata.mesh
    
    @property
    def cell_tags(self):
        return self.meshdata.cell_tags
    
    @property
    def facet_tags(self):
        return self.meshdata.facet_tags
    
    @property
    def ridge_tags(self):
        return self.meshdata.ridge_tags
    
    @property
    def peak_tags(self):
        return self.meshdata.peak_tags
    
    @property
    def Vs(self):
        if not hasattr(self, '_Vs'):
            raise NotImplementedError("Vs not implemented")
        return self._Vs
    
    @property
    def u_i(self):
        if not hasattr(self, '_u_i'): self._u_i = self.create_u_i()
        return self._u_i
    
    def create_u_i(self, prefix : str=''):
        u_i = [df.fem.Function(V) if V is not None else 0.0 for V in self.Vs]
        for i, u in enumerate(u_i):
            if hasattr(u, 'x'): u.x.array[:] = 0.0
            if hasattr(u, 'name'): u.name = prefix + 'u%'+repr(i)
        return u_i
    
    @property
    def u_a(self):
        if not hasattr(self, '_u_a'):
            self._u_a = [ufl.TrialFunction(V) if V is not None else None for V in self.Vs]
        return self._u_a
    
    @property
    def u_t(self):
        if not hasattr(self, '_u_t'):
            self._u_t = [ufl.TestFunction(V) if V is not None else None for V in self.Vs]
        return self._u_t
    
    @property
    def bcs(self):
        if not hasattr(self, '_bcs'):
            self._bcs = [[]]
        return self._bcs
    
    @property
    def ns(self):
        if not hasattr(self, '_ns'):
            self._ns = None
        return self._ns

    @property
    def F(self):
        if not hasattr(self, '_F'):
            raise NotImplementedError("F not implemented")
        return self._F
    
    @property
    def JF(self):
        if not hasattr(self, '_JF'):
            raise NotImplementedError("JF not implemented")
        return self._JF
    
    @property
    def Fvec(self):
        if not hasattr(self, '_Fvec'):
            raise NotImplementedError("Fvec not implemented")
        return self._Fvec
    
    @property
    def JFmat(self):
        if not hasattr(self, '_JFmat'):
            raise NotImplementedError("JFmat not implemented")
        return self._JFmat
    
    def formFunction(self, *args):
        raise NotImplementedError("formFunction not implemented")
    
    def formJacobian(self, *args):
        raise NotImplementedError("formJacobian not implemented")
    
    def solve(self, **kwargs):
        raise NotImplementedError("solve not implemented")

    @property
    def ds(self):
        if not hasattr(self, '_ds'):
            self._ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)
        return self._ds
    
    @property
    def dS(self):
        if not hasattr(self, '_dS'):
            self._dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.facet_tags)
        return self._dS
    
    @property
    def dx(self):
        if not hasattr(self, '_dx'):
            self._dx = ufl.Measure('dx', domain=self.mesh, subdomain_data=self.cell_tags)
        return self._dx
    
    @property
    def dr(self):
        if not hasattr(self, '_dr'):
            self._dr = ufl.Measure('dr', domain=self.mesh, subdomain_data=self.ridge_tags)
        return self._dr
    
    @property
    def dP(self):
        if not hasattr(self, '_dP'):
            self._dP = ufl.Measure('dP', domain=self.mesh, subdomain_data=self.peak_tags)
        return self._dP
    
    @property
    def u_vis(self):
        if not hasattr(self, '_u_vis'): self._u_vis = self.create_u_vis()
        return self._u_vis
    
    def create_u_vis(self, u_i : list=None, prefix : str=''):
        lu_i = self.u_i if u_i is None else u_i
        u_vis = []
        for i, u in enumerate(lu_i):
            if hasattr(u, 'function_space'):
                V = u.function_space
                mesh = V.mesh
                shape = V.value_shape
            else:
                mesh = self.mesh
                shape = ()
            if mesh.topology.dim == 0:
                _u_vis = df.fem.Function(df.fem.functionspace(mesh, ('DG', 0)+(shape,)))
            else:
                _u_vis = df.fem.Function(df.fem.functionspace(mesh, self.vis_e+(shape,)))
            if hasattr(u, 'name'):
                _u_vis.name = prefix + u.name
            else:
                _u_vis.name = prefix + 'u%'+repr(i)
            u_vis.append(_u_vis)
        return u_vis
    
    @property
    def vtxs(self):
        if not hasattr(self, '_vtxs'): self._vtxs = self.create_vtxs()
        return self._vtxs
    
    def create_vtxs(self, 
                    u_vis : list[df.fem.Function]=None, 
                    suffix : str=''):
        lu_vis = self.u_vis if u_vis is None else u_vis
        meshes = list(set([u.function_space.mesh for u in lu_vis]))
        meshnames = [mesh.name for mesh in meshes]
        vtxs = {mesh:None for mesh in meshes}
        if hasattr(self, 'output_basepath') and self.output_basepath is not None:
            filename = str(self.output_basepath) + suffix
            for m, mesh in enumerate(meshes):
                if len(vtxs) > 1:
                    meshname = mesh.name
                    if meshnames.count(meshname) > 1:
                        meshname = meshname + '%' + repr(meshnames[:m].count(meshname))
                    filename = str(self.output_basepath) + '_' + meshname + suffix
                u_vis_mesh = [u for u in lu_vis if u.function_space.mesh == mesh]
                if len(u_vis_mesh) > 0: vtxs[mesh] = df.io.VTXWriter(self.comm, filename+'.bp', u_vis_mesh)
        return vtxs

    def output(self, 
               vtxs : df.io.VTXWriter | dict[df.mesh.Mesh, df.io.VTXWriter]=None, 
               u_i : list=None, 
               u_vis : list[df.fem.Function]=None, 
               t : float=0.0):
        lvtxs = self.vtxs if vtxs is None else vtxs
        if not isinstance(lvtxs, dict): lvtxs = {self.mesh : vtxs}

        lu_vis = self.u_vis if u_vis is None else u_vis
        lu_i = self.u_i if u_i is None else u_i

        for u, v in zip(lu_i, lu_vis):
            if hasattr(u, 'function_space'): 
                v.interpolate(u)
            else:
                v.x.array[:] = u[0]
        
        for mesh, vtx in lvtxs.items():
            if vtx is not None:
                vtx.write(t)
    
#####################################################

class ProblemNest(Problem):
    """A base problem class assuming nested matrices and vectors"""
    @property
    def Fvec(self):
        if not hasattr(self, '_Fvec'):
            self._Fvec = df.cpp.fem.petsc.create_vector_nest([(V.dofmap.index_map, V.dofmap.index_map_bs) for V in self.Vs])
        return self._Fvec
    
    @property
    def JFmat(self):
        if not hasattr(self, '_JFmat'):
            self._JFmat = df.fem.petsc.create_matrix(df.fem.form(self.JF), kind=PETSc.Mat.Type.NEST)
        return self._JFmat
    
    @property
    def x(self):
        if not hasattr(self, '_x'):
            self._x = PETSc.Vec().createNest([df.la.petsc.create_vector_wrap(u.x) for u in self.u_i], 
                                             comm=self.comm)
        return self._x
    
    def Vec2Function(self, x, u_i):
        for i, (x_sub, u) in enumerate(zip(x.getNestSubVecs(), u_i)):
            if hasattr(u, 'x'):
                x_sub.copy(u.x.petsc_vec)
                u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            else:
                u_i[i][0] = x_sub.getValue(0) # assume scalar
        
#####################################################

class SNESProblemNest(ProblemNest):
    """A problem class for solving nonlinear problems with SNES assuming nested matrices and vectors."""
    def reset(self):
        super().reset()
        for attr in [
                    '_snes', '_snes_vtxs', '_snes_vtx_i', '_snes_u_vis',
                    ]:
            if hasattr(self, attr): delattr(self, attr)

    def __del__(self):
        super().__del__()
        if hasattr(self, '_snes_vtxs'): 
            for vtx in self._snes_vtxs.values(): vtx.close()

    @property
    def JF(self):
        if not hasattr(self, '_JF'):
            F = self.F
            self._JF = [[ufl.derivative(F[i], self.u_i[j], self.u_a[j]) if F[i] is not None and self.u_a[j] is not None else None for j in range(len(self.u_i))] 
                         for i in range(len(F))]
        return self._JF

    def formFunction(self, snes, x, f):
        self.Vec2Function(x, self.u_i)

        for f_sub in f.getNestSubVecs():
            with f_sub.localForm() as f_sub_loc: f_sub_loc.set(0.0)
        f = df.fem.petsc.assemble_vector(f, df.fem.form(self.F))
        for f_sub in f.getNestSubVecs():
            f_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.petsc.set_bc(f, self.bcs, alpha=0.0)

    def formJacobian(self, snes, x, A, B):
        self.Vec2Function(x, self.u_i)

        A.zeroEntries()
        A = df.fem.petsc.assemble_matrix(A, df.fem.form(self.JF), bcs=[bc for bcs in self.bcs for bc in bcs])
        A.assemble()

        if self.ns is not None:
            assert(self.ns.test(A))
            A.setNullSpace(self.ns)

        return True # same non-zero pattern
    
    @property
    def snes_u_vis(self):
        if not hasattr(self, '_snes_u_vis'): 
            self._snes_u_vis = []
            # the order of prefixes here has to match snesMonitor
            for prefix in ['iterated_', 'update_', 'residual_']:
                self._snes_u_vis += self.create_u_vis(prefix=prefix)
        return self._snes_u_vis
    
    @property
    def snes_vtxs(self):
        if not hasattr(self, '_snes_vtxs'): 
            self._snes_vtxs = self.create_vtxs(u_vis=self.snes_u_vis, suffix='_snes')
        return self._snes_vtxs
    
    @property
    def snes_vtx_i(self):
        if not hasattr(self, '_snes_vtx_i'): self._snes_vtx_i = 0
        return self._snes_vtx_i
    
    @snes_vtx_i.setter
    def snes_vtx_i(self, i):
        self._snes_vtx_i = i
    
    @property
    def snes(self):
        if not hasattr(self, '_snes'):
            self._snes = PETSc.SNES().create(comm=self.comm)
            self._snes.setFunction(self.formFunction, self.Fvec)
            self._snes.setJacobian(self.formJacobian, self.JFmat)

            self._snes.setType(self._snes.Type.NEWTONLS)

            ksp = self._snes.getKSP()
            ksp.setType(ksp.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(pc.Type.LU)
            pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

            self._snes.setFromOptions()
        return self._snes
    
    def snesMonitor(self, snes, iter, fnorm):
        print('in snesMonitor')
        # import ipdb; ipdb.set_trace()
        u_i = self.create_u_i(prefix='iterated_')
        self.Vec2Function(snes.getSolution(), u_i)
        du_i = self.create_u_i(prefix='update_')
        self.Vec2Function(snes.getSolutionUpdate(), du_i)
        r_i = self.create_u_i(prefix='residual_')
        self.Vec2Function(snes.getFunction()[0], r_i)

        self.output(self.snes_vtxs, u_i+du_i+r_i, self.snes_u_vis, iter)
        self.snes_vtx_i += 1
    
    def reset_snes(self):
        if hasattr(self, '_snes'):
            self._snes.destroy()
            delattr(self, '_snes')

    def solve(self, petsc_options={}, monitor : bool=False):
        self.reset_snes()

        opts = PETSc.Options()
        for k,v in petsc_options.items(): opts[k] = v

        for u, bcs in zip(self.u_i, self.bcs):
            for bc in bcs:
                bc.set(u.x.array)

        if monitor: self.snes.setMonitor(self.snesMonitor)

        self.output(t=0.0)
        self.snes.solve(x=self.x)
        self.output(t=1.0)
    
#####################################################

class TSProblemNest(SNESProblemNest):
    """A problem class for solving time-dependent nonlinear problems with TS assuming nested matrices and vectors."""

    def reset(self):
        super().reset()
        for attr in [
                    '_u_dot',
                    '_a_c',
                    '_ts',
                    '_ts_step_stage_i',
                    ]:
            if hasattr(self, attr): delattr(self, attr)

    @property
    def a_c(self):
        if not hasattr(self, '_a_c'):
            self._a_c = df.fem.Constant(self.mesh, df.default_scalar_type(-666.0))
        return self._a_c
    
    @property
    def u_dot(self):
        if not hasattr(self, '_u_dot'): self._u_dot = self.create_u_i(prefix='dot_')
        return self._u_dot
    
    @property
    def JF(self):
        if not hasattr(self, '_JF'):
            F = self.F
            self._JF = [[self.a_c*ufl.derivative(F[i], self.u_dot[j], self.u_a[j]) + ufl.derivative(F[i], self.u_i[j], self.u_a[j]) if F[i] is not None and self.u_a[j] is not None else None for j in range(len(self.u_i))] 
                       for i in range(len(F))]
        return self._JF

    def formFunction(self, snes, x, f):
        raise NotImplementedError("formFunction not implemented")
    
    def formJacobian(self, snes, x, A, B):
        raise NotImplementedError("formJacobian not implemented")

    def formIFunction(self, ts, t, x, xdot, f):
        self.Vec2Function(x, self.u_i)
        self.Vec2Function(xdot, self.u_dot)

        for f_sub in f.getNestSubVecs():
            with f_sub.localForm() as f_sub_loc: f_sub_loc.set(0.0)
        f = df.fem.petsc.assemble_vector(f, df.fem.form(self.F))
        for f_sub in f.getNestSubVecs():
            f_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.petsc.set_bc(f, self.bcs, alpha=0.0)

    def formIJacobian(self, ts, t, x, xdot, a, A, B):
        self.Vec2Function(x, self.u_i)
        self.Vec2Function(xdot, self.u_dot)

        self.a_c.value = a

        A.zeroEntries()
        A = df.fem.petsc.assemble_matrix(A, df.fem.form(self.JF), bcs=[bc for bcs in self.bcs for bc in bcs])
        A.assemble()

        if self.ns is not None:
            assert(self.ns.test(A))
            A.setNullSpace(self.ns)
        
        return True # same non-zero pattern
    
    @property
    def ts_step_stage_i(self):
        if not hasattr(self, '_ts_step_stage_i'): self._ts_step_stage_i = 0
        return self._ts_step_stage_i
    
    @ts_step_stage_i.setter
    def ts_step_stage_i(self, i):
        self._ts_step_stage_i = i

    def tsPreStep(self, ts):
        #print('in tsPreStep', flush=True)
        self.ts_step_stage_i = -1
        self.snes_vtx_i = 0
        # FIXME: the following should happen in a prestage callback but
        # petsc4py doesn't seem to expose that so here we assume a single
        # stage method and do the setup in the prestep
        # this renders ts_step_stage_i unnecessary but it's been left it for
        # a future fix
        if hasattr(self, 'snes_monitor') and self.snes_monitor:
            ti = ts.getStepNumber()
            self.ts_step_stage_i += 1
            # setup the _snes_vtxs attribute so that the file exists with the right filename
            if hasattr(self, '_snes_vtxs'):
                for vtx in self._snes_vtxs.values(): 
                    if vtx is not None: vtx.close()
            self._snes_vtxs = self.create_vtxs(u_vis=self.snes_u_vis, 
                                               suffix='_ts_{:d}_{:d}_snes'.format(ti, self.ts_step_stage_i))

    def tsPostStep(self, ts):
        t = ts.getTime()
        x = ts.getSolution()
        self.Vec2Function(x, self.u_i)
        self.output(t=t)

    @property
    def ts(self):
        if not hasattr(self, '_ts'):
            self._ts = PETSc.TS().create(comm=self.comm)
            self._ts.setIFunction(self.formIFunction, self.Fvec)
            self._ts.setIJacobian(self.formIJacobian, self.JFmat)
            self._ts.setPreStep(self.tsPreStep)
            # self._ts.setPreStage(self.tsPreStage) # not available in petsc4py
            self._ts.setPostStep(self.tsPostStep)

            self._ts.setType(self._ts.Type.BDF)

            self._ts.setTime(0.0)
            self._ts.setTimeStep(1.e-4)
            self._ts.setMaxTime(1)
            self._ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
            self._ts.setMaxSNESFailures(-1)

            snes = self._ts.getSNES()
            snes.setTolerances(max_it=10)
            ksp = snes.getKSP()
            ksp.setType(ksp.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(pc.Type.LU)
            pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

            self._ts.setFromOptions()
        return self._ts
    
    @property
    def snes(self):
        if not hasattr(self, '_snes'):
            self._snes = self.ts.getSNES()
        return self._snes

    def reset_ts(self):
        if hasattr(self, '_ts'):
            if hasattr(self, '_snes'): delattr(self, '_snes')
            self._ts.destroy()
            delattr(self, '_ts')
    
    def solve(self, petsc_options={}, snes_monitor : bool=False):
        self.reset_ts()
        self.snes_monitor = snes_monitor

        opts = PETSc.Options()
        for k,v in petsc_options.items(): opts[k] = v

        for u, bcs in zip(self.u_i, self.bcs):
            for bc in bcs:
                bc.set(u.x.array)
        
        if self.snes_monitor: self.snes.setMonitor(self.snesMonitor)

        self.output(t=0.0)
        self.ts.solve(self.x)
