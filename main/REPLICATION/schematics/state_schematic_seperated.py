import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
cmap = plt.cm.viridis
from matplotlib import gridspec

def make_axes(rows=3, cols=4, figs=1, figsize=(15,12)):
    figlist = []
    for _ in range(figs):
        fig, axs = plt.subplots(rows, cols, figsize=figsize, tight_layout=True)
        for row in range(rows):
            fig.delaxes(axs[row, 3])
            axs[row, 3] = fig.add_subplot(rows, cols, cols * row + 4, projection='3d')
        figlist.append(axs)
    return figlist

def draw_embedding_panel(ax, vecs, ctx, lbl, col, version):
    if ctx == 1: ax.set_title(f"{lbl} Embeddings", fontsize=20, fontweight='bold', color=col)
    ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], aspect='equal'); ax.axis('off')
    for i, vec in enumerate(vecs):
        is_ctx = i < ctx
        fc, lw, a = (col, 5, 1) if is_ctx else (col, 2, 0.2)
        v = vec * (1.2 if is_ctx and version == 0 else 1)
        ax.arrow(0, 0, *v, head_width=0.1*(version==0), head_length=0.1*(version==0),
                 fc=fc, ec=fc, lw=lw, alpha=a, zorder=5 if is_ctx else -10)
        if is_ctx:
            ax.text(*(v * (1.3 if version==0 else 1.4)), rf"{lbl}$_{i}$", color=col, fontsize=15, weight='bold',
                    ha='center', va='center')
            if version == 1:
                ax.arrow(0, 0, *vec*1.02, head_width=0.1, head_length=0.1, fc=fc, ec=fc, lw=lw, alpha=a, zorder=5)
    if version == 1:
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls='--', color=col, alpha=0.5))

def draw_interaction_panel(ax, vK, vQ, ctx, r):
    if r == 0: ax.set_title("Interactions\n" + r"Z$_{cc'} = $K$_c^T $Q$_{c'}$", fontsize=20, color='purple', weight='bold')
    ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], aspect='equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls='--', color='gray', alpha=0.3))
    t = np.linspace(0, 1, 50)
    for c in range(ctx):
        K, Q = vK[c], vQ[c]
        ax.arrow(0, 0, *K, fc='C0', ec='C0', lw=5, head_width=0.1, head_length=0.1)
        ax.arrow(0, 0, *Q, fc='C1', ec='C1', lw=5, head_width=0.1, head_length=0.1)
        for d in range(ctx):
            if r == 0 or c != d:
                Qd = vQ[d]
                mid = (K + Qd) / 2
                delta = Qd - K
                perp = np.array([-delta[1], delta[0]])
                perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else np.array([0,1])
                if perp.dot(mid) < 0: perp = -perp
                ctrl = mid + perp * 0.5
                bez = np.outer((1 - t) ** 2, K) + np.outer(2 * (1 - t) * t, ctrl) + np.outer(t ** 2, Qd)
                ax.plot(bez[:, 0], bez[:, 1], '--', color='purple', lw=2)
                pos = ctrl + perp * 0.05
                ax.text(*pos, f"Z$_{{{c},{d}}}$", fontsize=15, color='purple', ha='center', va='center', weight='bold')

def plot_1d_panel(ax, vec, norm):
    x = np.arange(len(vec))
    z = 1 + (vec - vec.min()) / (vec.max() - vec.min()) * 0.3
    ax.cla(); minz = z.min(); below = minz - 0.07 * (z.max() - z.min())
    ax.plot(x, [0]*len(x), z, lw=4, color=cmap(0.6))
    for i in range(len(x) - 1):
        c = cmap(norm((vec[i] + vec[i + 1]) / 2))
        ax.plot(x[i:i + 2], [0, 0], z[i:i + 2], lw=4, color=c)
    ax.plot([x[0], x[-1]], [0, 0], [below, below], color='k', lw=1)
    ax.text((x[0]+x[-1])/2, -0.04, below, r'R$_0$', color = 'g', ha='center', va='top', fontsize=17, weight='bold')
    ax.view_init(elev=0, azim=90)
    ax.set_box_aspect([2.5, 1, 1]); ax.axis('off')

def plot_2d_panel(ax, mat):
    face_vals = 0.25 * (mat[:-1, :-1] + mat[1:, :-1] + mat[:-1, 1:] + mat[1:, 1:])
    X, Y = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
    Z2 = 1 + (mat - mat.min()) / (mat.max() - mat.min()) * 0.05
    norm2 = Normalize(vmin=mat.min(), vmax=mat.max())
    ax.cla()
    ax.plot_surface(X, Y, Z2, facecolors=cmap(norm2(face_vals)), rstride=1, cstride=1,
                    shade=False, linewidth=0.3, antialiased=True)
    ax.view_init(elev=45, azim=-75)
    ax.text(9.5, -1, 1, r'R$_0$', color='g', ha='center', va='center', fontsize=17, weight='bold')
    ax.text(21,12, 1, r'R$_1$', color='g', ha='center', va='center', fontsize=17, weight='bold', rotation=90)
    ax.set_box_aspect([1.2,1,0.5]); ax.axis('off')

def plot_3d_cube(ax, N=10, sizes=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,1]):
    u = np.linspace(0,1,N); U,V=np.meshgrid(u,u)
    faces=['front','back','left','right','top','bottom']
    raw={f: gaussian_filter(np.random.rand(N,N), sigma=1) for f in faces}
    ax.cla()
    for s in sizes:
        amp, ec, alpha = (.05,'k',.2) if s>.95 else (.1,'none',.2) if s>.8 else (1*(1-s),'none',.5*(1-s))
        norm3=Normalize(vmin=min(raw[f].min() for f in faces), vmax=max(raw[f].max() for f in faces))
        start,end=(1-s)/2,(1-s)/2+s
        for face in faces:
            disp=(raw[face]-0.5)*2*amp; color=cmap(norm3(raw[face]))
            s_min_d,e_plus_d=start-disp,end+disp
            if face=='front': ax.plot_surface(start+U*s,start+V*s,e_plus_d,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
            elif face=='back': ax.plot_surface(start+U[::-1]*s,start+V*s,s_min_d,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
            elif face=='left': ax.plot_surface(s_min_d,start+U*s,start+V*s,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
            elif face=='right': ax.plot_surface(e_plus_d,start+U*s,start+V*s,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
            elif face=='top': ax.plot_surface(start+U*s,e_plus_d,start+V*s,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
            elif face=='bottom': ax.plot_surface(start+U*s,s_min_d,start+V[::-1]*s,facecolors=color,shade=False,alpha=alpha,edgecolor=ec)
    ax.set_box_aspect([1,1,1])
    ax.text(1,0.8,-0.3,r'R$_0$',color='g',fontsize=17,weight='bold',ha='left',va='bottom')
    ax.text(1,1.2,.3,r'R$_1$',color='g',fontsize=17,weight='bold',ha='left',va='bottom')
    ax.text(.4,1.1,1,r'R$_2$',color='g',fontsize=17,weight='bold',ha='left',va='bottom')
    ax.axis('off')

def make_embeddings(version=0, num_states=60):
    aK,aQ=np.random.rand(2,num_states)*2*np.pi; figs=make_axes()
    for r,ctx in enumerate(range(1,4)):
        vK,vQ=np.c_[np.cos(aK),np.sin(aK)],np.c_[np.cos(aQ),np.sin(aQ)]
        for axs in figs:
            draw_embedding_panel(axs[r,0],vK,ctx,'K','C0',version)
            axs[r,0].text(-2.5, 0, rf"$|C|={ctx}$", fontsize=20, fontweight='bold', color='black',
                          va='center', ha='right')
            draw_embedding_panel(axs[r,1],vQ,ctx,'Q','C1',version)
            draw_interaction_panel(axs[r,2],vK,vQ,ctx,r)
    return figs

def fill_manifolds(axs):
    ax1,ax2,ax3=axs[0,3],axs[1,3],axs[2,3]
    vec = np.sin(2 * np.pi * np.linspace(np.random.rand() * .8, .8 + .5 * np.random.rand(), 40))
    mat=gaussian_filter(np.random.rand(20,20),sigma=3); mat/=mat.sum()
    tensor=np.empty((10,10,10))
    for k in range(10): t=gaussian_filter(np.random.rand(10,10),sigma=2); t/=t.sum(); tensor[:,:,k]=t
    all_vals=np.concatenate([vec,mat.flatten(),tensor.flatten()])
    norm=Normalize(vmin=all_vals.min(),vmax=all_vals.max())
    plot_1d_panel(ax1,vec,norm); plot_2d_panel(ax2,mat); plot_3d_cube(ax3)
    ax1.set_title("Likelihoods\n"+r"$P(obs=0|$R$_0,...$R$_{|C|})$",color='g',fontsize=20,fontweight='bold')

if __name__ == "__main__":
    figs = make_embeddings(version=1)
    for axs in figs: fill_manifolds(axs)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge, Polygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, **kw):
        super().__init__((0,0),(0,0), **kw)
        self._verts3d = (xs, ys, zs)
    def draw(self, renderer):
        xs, ys, _ = proj3d.proj_transform(*self._verts3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
    def do_3d_projection(self, renderer=None):
        xs, ys, _ = proj3d.proj_transform(*self._verts3d, self.axes.M)
        self.set_positions((xs[1], ys[1]), (xs[0], ys[0]))
        return np.inf

def plot_circle(ax, angle=None):
    Z = np.deg2rad(angle if angle is not None else np.random.randint(60,180))
    θ = np.linspace(0, 2*np.pi, 1000)
    r = 1 + .13 + .05*np.sin(10*θ)  # slightly reduced amplitude
    x, y = r*np.cos(θ), r*np.sin(θ)
    mask = np.abs((θ - Z + np.pi) % (2*np.pi) - np.pi) < np.pi/6

    ax.add_patch(Wedge((0,0), 1, 0, np.rad2deg(Z), facecolor='purple', alpha=.05))
    ax.add_patch(Wedge((0,0), .3, 0, np.rad2deg(Z), facecolor='purple', alpha=.1))
    ax.plot(np.where(mask, np.nan, x), np.where(mask, np.nan, y), '-', c='purple', lw=1, alpha=.5)
    ax.plot(x[mask], y[mask], c='purple', lw=3)
    i0, i1 = np.where(mask)[0][[0, -1]]
    ax.plot([x[i0], x[i1]], [y[i0], y[i1]], 'sk', ms=6)
    ax.plot([0, 1], [0, 0], '--', c='purple', lw=1)
    ax.plot([0, np.cos(Z)], [0, np.sin(Z)], c='purple', lw=2)
    ax.plot(.3*np.cos(np.linspace(0, Z, 100)), .3*np.sin(np.linspace(0, Z, 100)), c='purple', lw=2)
    ax.add_patch(Circle((0,0), 1, fill=False, ls='--', color='gray', alpha=.3))

    ax.text(0.25, .3, r'Z$_{0,0}$', fontsize=12, fontweight='bold', color='purple')  # moved closer to arc
    x_mu, y_mu = x[mask].mean(), y[mask].mean()
    x_loc = .6 * x_mu if x_mu > 0 else -.5 + 1.2 * x_mu
    y_loc = y_mu + .15  # more consistent top spacing
    ax.text(x_loc, y_loc, r'V$_{0,0}$', fontsize=14, fontweight='bold', color='purple')

    ax.set(aspect='equal', xlim=[-1.45, 1.45], ylim=[-1.45, 1.45])
    ax.axis('off')
    return x[mask], y[mask]

fig = plt.figure(figsize=(10,8))
gs = GridSpec(5,1, height_ratios=[1,.1,.9,.1,1.2], hspace=0.01)

# Top circle
ax1 = fig.add_subplot(gs[0])
plot_circle(ax1)

# Middle diamond + arrows + sine
ax2 = fig.add_subplot(gs[2])
u, v = np.array([-1,-1]), np.array([1,-1])
pts = np.vstack(((0,0), u, u+v, v))
ax2.add_patch(Polygon(pts, facecolor='#e0e7ef', alpha=.4, edgecolor='none'))
for P, Q in (((0,0),u), (u,u+v), (u+v,v), (v,(0,0))):
    ax2.plot(*zip(P,Q), '--', c='k', lw=1)
for vec, shiftx, shifty in zip((u, v), (-.15, .15), (.15, .15)):
    arr = FancyArrowPatch((0,0), tuple(vec), arrowstyle='-|>', mutation_scale=30, lw=0, color='purple', zorder=1e3)
    ax2.add_patch(arr)
    L = np.linalg.norm(vec); d = vec/L; perp = np.array([-d[1], d[0]])
    t = np.linspace(0, .85*L, 200)
    w = .04*np.sin((.5+np.random.rand()*.5)*np.pi*4*(t/L))
    x = d[0]*t + perp[0]*w + shiftx
    y =  d[1]*t + perp[1]*w + shifty

    x -= x[0]
    y -= y[0]
    ax2.plot(x,y, c='purple', lw=3)
    ax2.text(-1.05, -.35, r'V$_{1,0}$', fontsize=14, fontweight='bold', color='purple')  # 
    ax2.text(0.35, -.35, r'V$_{0,1}$', fontsize=14, fontweight='bold', color='purple')   # 
    ax2.plot(x[0],y[0],'sk', ms=6)

ax2.set(aspect='equal'); ax2.axis('off')

# Bottom cube with hidden-face culling
ax3 = fig.add_subplot(gs[4], projection='3d')
faces = np.array([
    [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
    [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
    [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
    [[0,1,0],[1,1,0],[1,1,1],[0,1,1]],
    [[0,0,0],[0,1,0],[0,1,1],[0,0,1]],
    [[1,0,0],[1,1,0],[1,1,1],[1,0,1]]
])
base = np.array(to_rgb('#e0e7ef'))
shade = np.array([1,1,.5,1,.8,1])
visible_idxs = [0, 2, 4]
for i in visible_idxs:
    fc = base * shade[i]
    ax3.add_collection3d(Poly3DCollection([faces[i]], facecolors=[fc], edgecolors='none', alpha=1))

edges = np.array([
    [[1,0,0],[1,1,0]], [[1,1,0],[0,1,0]], [[0,1,0],[0,1,1]], [[0,1,1],[0,0,1]],
    [[0,0,1],[1,0,1]], [[1,0,1],[1,1,1]], [[1,1,1],[1,1,0]], [[1,1,1],[0,1,1]]
])
mask_top  = (edges[:,0,2]==1)&(edges[:,1,2]==1)
mask_back = (edges[:,0,2]==0)&(edges[:,1,2]==0)
ax3.add_collection3d(Line3DCollection(edges[mask_top], colors='k', linestyles='dashed', linewidths=1))
ax3.add_collection3d(Line3DCollection(edges[mask_back], colors='k', linestyles='dashed', linewidths=2, zorder=100))
ax3.add_collection3d(Line3DCollection(edges[~(mask_top|mask_back)], colors='k', linestyles='solid', linewidths=1))

dashed_manual = [
    [(0,0,0),(0,1,0)], [(0,1,1),(0,1,0)], [(0,0,0),(0,0,1)],
    [(0,0,0),(1,0,0)], [(1,0,0),(1,0,1)]
]
ax3.add_collection3d(Line3DCollection(dashed_manual, colors='k', linestyles='dashed', linewidths=1, zorder=100))

ax3.view_init(elev=30, azim=220)
ax3.set_box_aspect([1,1,1])
ax3.set_axis_off()

# overlay arrows + sine-wave decoration
overlay = fig.add_axes(ax3.get_position(), frameon=False)
overlay.patch.set_alpha(0)
overlay.set(xticks=[], yticks=[])
x_offsets = [[.015, -.015],[-.025,-.005],[.005,.02]]
y_offsets = [[.025, .025],[.0025,-.025],[-.025,.0025]]
for x_offset, y_offset, vec in zip(x_offsets, y_offsets, ((0,0,0),(1,0,1),(0,1,1))):
    # 2D arrow
    x0, y0, _ = proj3d.proj_transform(0,0,1, ax3.get_proj())
    x1, y1, _ = proj3d.proj_transform(*vec, ax3.get_proj())
    d0 = ax3.transData.transform((x0,y0))
    d1 = ax3.transData.transform((x1,y1))
    f0 = fig.transFigure.inverted().transform(d0)
    f1 = fig.transFigure.inverted().transform(d1)
    overlay.add_patch(FancyArrowPatch(f0, f1 , transform=fig.transFigure,
                                     arrowstyle='-|>', mutation_scale=40,
                                     lw=0, color='purple', zorder=1000))
    # sine-wave overlay
    for s in [1,-1]:
        vec2 = np.array(f1) - np.array(f0)
        L2 = np.linalg.norm(vec2)
        u2 = vec2 / L2
        perp2 = np.array([-u2[1], u2[0]])
        t2 = np.linspace(0, .85 * L2, 200)
        w2 = .004 * np.sin( 2.5 * np.pi*(t2/L2) +   (np.random.rand() + .1) *  s * np.pi)
        xs = u2[0]*t2 + perp2[0]*w2 
        xs -= xs[0]
        xs += f0[0]

        ys = u2[1]*t2 + perp2[1]*w2
        ys -= ys[0]
        ys +=  f0[1] 

        lw = Line2D(xs, ys, transform=fig.transFigure, color='purple', linewidth=3, zorder=1, alpha = .7)
        fig.add_artist(lw)
        lw = Line2D(xs, ys, transform=fig.transFigure, color='purple', linewidth=1, zorder=1)        
        fig.add_artist(lw)

        sct = Line2D([xs[0]],[ys[0]], marker ='s', color = 'k', ms=5, transform=fig.transFigure, zorder=999)
        fig.add_artist(sct)

cube_labels = [
    ((.25, .1, 1.22), r'V$_{2,1}$'),
    ((-.22, .4, 1.22), r'V$_{1,2}$'),
    ((.82, .28, .52), r'V$_{2,0}$'),
    ((.2, 1.02, .52), r'V$_{1,0}$'),
    ((.5, .25, .02), r'V$_{0,2}$'),
    ((.2, .65, .02), r'V$_{0,1}$')
]
for pos, label in cube_labels:
    ax3.text(*pos, label, fontsize=12, fontweight='bold', color='purple', zorder=999)

plt.show()
