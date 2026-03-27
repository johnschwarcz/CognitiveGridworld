import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from scipy.ndimage import gaussian_filter
from matplotlib.patches import FancyArrowPatch, Wedge, Polygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D
cmap = plt.cm.viridis

from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns 
sns.set()
sns.set_style("white")

def make_axes(rows=3, cols=5, figs=1, figsize=(25,12)):
    figlist = []
    for _ in range(figs):
        fig, axs = plt.subplots(rows, cols, figsize=figsize, tight_layout=True)
        for row in range(rows):
            fig.delaxes(axs[row, 4])  
            axs[row, 4] = fig.add_subplot(rows, cols, cols * row + 5, projection='3d')
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

def draw_interaction_panel(ax, vK, vQ, ctx, r, K_col, Q_col, Z_col):
    # if r == 0: ax.set_title("Interactions\n" + r"Z$_{cc'} = $K$_c^T $Q$_{c'}$", fontsize=20, color=Z_col, weight='bold')
    if r == 0: ax.set_title("Compression", fontsize=20, color=Z_col, weight='bold')
    ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], aspect='equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls='--', color='gray', alpha=0.5))
    t = np.linspace(0, 1, 50)
    for c in range(ctx):
        K, Q = vK[c], vQ[c]
        ax.arrow(0, 0, *K, fc=K_col, ec=K_col, lw=5, head_width=0.1, head_length=0.1)
        ax.arrow(0, 0, *Q, fc=Q_col, ec=Q_col, lw=5, head_width=0.1, head_length=0.1)
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
                ax.plot(bez[:, 0], bez[:, 1], '--', color=Z_col, lw=2)
                pos = ctrl + perp * 0.05
                ax.text(*pos, f"Z$_{{{c},{d}}}$", fontsize=15, color=Z_col, ha='center', va='center', weight='bold')

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
    K_col = 'C0'#'steelblue'
    Q_col = 'C1'#'midnightblue'#
    Z_col ='purple'# "#BB47B5"
    aK,aQ=np.random.rand(2,num_states)*2*np.pi; figs=make_axes()
    allK, allQ = [np.zeros(3, dtype = object) for _ in range(2)]
    aQ[0] = min(np.pi * .9, max(np.pi/3, aQ[0]/2))
    aK[0] = 0 
    for r,ctx in enumerate(range(1,4)):
        vK,vQ=np.c_[np.cos(aK),np.sin(aK)],np.c_[np.cos(aQ),np.sin(aQ)]
        allK[r-1] = vK 
        allQ[r-1] = vQ
        for axs in figs:
            draw_embedding_panel(axs[r,0],vK,ctx,'K',K_col,version)
            # axs[r,0].text(-2.5, 0, rf"$|C|={ctx}$", fontsize=20, fontweight='bold', color='black',
            #               va='center', ha='right')
            draw_embedding_panel(axs[r,1],vQ,ctx,'Q',Q_col,version)
            draw_interaction_panel(axs[r,2],vK,vQ,ctx,r, K_col, Q_col, Z_col)
    return figs, allK, allQ


def plot_circle(ax, V_col, angle=None):
    Z = np.deg2rad(angle if angle is not None else np.random.randint(60,180))
    θ = np.linspace(0, 2*np.pi, 1000)
    r = 1 + .14 + .08*np.sin(8*θ)  # slightly reduced amplitude
    x, y = r*np.cos(θ), r*np.sin(θ)
    mask = np.abs((θ - Z + np.pi) % (2*np.pi) - np.pi) < .1 * np.pi

    ax.add_patch(Wedge((0,0), 1, 0, np.rad2deg(Z), facecolor=V_col, alpha=.05))
    ax.add_patch(Wedge((0,0), .3, 0, np.rad2deg(Z), facecolor=V_col, alpha=.1))

    ax.plot(np.where(mask, np.nan, x), np.where(mask, np.nan, y), '-', c=V_col, lw=1, alpha=.5)
    ax.plot(x[mask], y[mask], c=V_col, lw=3)

    i0, i1 = np.where(mask)[0][[0, -1]]
    ax.plot([x[i0], x[i1]], [y[i0], y[i1]], 'sk', ms=6)
    ax.plot([0, 1], [0, 0], '--', c=V_col, lw=1)
    ax.plot([0, np.cos(Z)], [0, np.sin(Z)], c=V_col, lw=2)
    ax.plot(.3*np.cos(np.linspace(0, Z, 100)), .3*np.sin(np.linspace(0, Z, 100)), c=V_col, lw=2)
    ax.add_patch(Circle((0,0), 1, fill=False, ls='--', color='gray', alpha=.5))

    ax.text(0.25, .3, r'Z$_{0,0}$', fontsize=12, fontweight='bold', color=V_col)  # moved closer to arc
    x_mu, y_mu = x[mask].mean(), y[mask].mean()
    x_loc = .6 * x_mu if x_mu > 0 else -.5 + 1.2 * x_mu
    y_loc = y_mu + .15  # more consistent top spacing
    ax.text(x_loc, y_loc, r'V$_{0,0}$', fontsize=14, fontweight='bold', color=V_col)

    ax.set(aspect='equal', xlim=[-1.45, 1.45], ylim=[-1.45, 1.45])
    ax.axis('off')
    return r[mask]

def fill_manifolds(axs, V_col, rvec):
    ax1, ax2, ax3 = axs[0,4], axs[1,4], axs[2,4]

    vec = np.sin(2 * np.pi * np.linspace(np.random.rand() * .8, .8 + .5 * np.random.rand(), 40))
    mat=gaussian_filter(np.random.rand(20,20),sigma=3); mat/=mat.sum()
    tensor=np.empty((10,10,10))
    for k in range(10): t=gaussian_filter(np.random.rand(10,10),sigma=2); t/=t.sum(); tensor[:,:,k]=t
    all_vals=np.concatenate([vec,mat.flatten(),tensor.flatten()])
    # norm=Normalize(vmin=all_vals.min(),vmax=all_vals.max())

    norm = Normalize(vmin= rvec.min(), vmax= rvec.max())
    plot_1d_panel(ax1,rvec,norm)
    plot_2d_panel(ax2,mat)
    plot_3d_cube(ax3)
    # ax1.set_title("Likelihoods\n"+r"$P(obs=0|$R$_0,...$R$_{|C|})$",color='g',fontsize=20,fontweight='bold')
    ax1.set_title("Likelihoods",color='g',fontsize=20,fontweight='bold')

def plot_polygon_expansion(ax, V_col, scale = .8):
    u, v = np.array([-1, -1]), np.array([1, -1])
    pts = np.vstack(((0, 0), u * scale, (u + v) * scale, v * scale))
    # ax.add_patch(Polygon(pts, facecolor="#a0c8f5", alpha=.1, edgecolor='none'))
    ax.add_patch(Polygon(pts, facecolor="purple", alpha=.05, edgecolor='none'))

    for P, Q in (((0, 0), u * scale), (u * scale, u * scale + v * scale),\
                (u * scale + v * scale, v * scale), (v * scale, (0, 0))):
        ax.plot(*zip(P, Q), '--', c='k', lw=2)

    for vec, shiftx, shifty in zip((u, v), (-.15, .15), (.15, .15)):
        arr = FancyArrowPatch((0, 0), tuple(vec * .95), arrowstyle='-|>', mutation_scale=80, lw=0,
                              color=V_col, zorder=1e3)
        ax.add_patch(arr)
        L = np.linalg.norm(vec)
        d = vec / L
        perp = np.array([-d[1], d[0]])
        t = np.linspace(0, .85 * L, 200)
        w = .04 * np.sin((.5 + np.random.rand() * .5) * np.pi * 4 * (t / L))
        x = d[0] * t + perp[0] * w + shiftx
        y = d[1] * t + perp[1] * w + shifty
        x -= x[0]
        y -= y[0]
        ax.plot(x, y, c=V_col, lw=4)
        ax.plot(x[0], y[0], 'sk', ms=10)

    ax.text(-1, -.35, r'V$_{1,0}$', fontsize=14, fontweight='bold', color=V_col)
    ax.text( 0.55, -.35, r'V$_{0,1}$', fontsize=14, fontweight='bold', color=V_col)
    ax.set(aspect='equal')
    ax.axis('off')

def unit_vec(p1, p2):
    v = np.array(p2) - np.array(p1)
    return v / np.linalg.norm(v)
    
def plot_cube_expansion(ax, fig, V_col):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.colors import to_rgb

    # --- Draw cube ---
    faces = np.array([
        [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
        [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
        [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
        [[0,1,0],[1,1,0],[1,1,1],[0,1,1]],
        [[0,0,0],[0,1,0],[0,1,1],[0,0,1]],
        [[1,0,0],[1,1,0],[1,1,1],[1,0,1]]
    ])
    # base = np.array(to_rgb("#deeeff"))
    base = np.array(to_rgb("#fef3ff"))
    shade = np.array([1,1,.8,1,.9,1])
    visible_idxs = [0, 2, 4]
    for i in visible_idxs:
        ax.add_collection3d(Poly3DCollection([faces[i]], facecolors=[base * shade[i]], edgecolors='none'))

    edges = np.array([
        [[1,0,0],[1,1,0]], [[1,1,0],[0,1,0]], [[0,1,0],[0,1,1]], [[0,1,1],[0,0,1]],
        [[0,0,1],[1,0,1]], [[1,0,1],[1,1,1]], [[1,1,1],[1,1,0]], [[1,1,1],[0,1,1]]
    ])
    mask_top  = (edges[:,0,2]==1)&(edges[:,1,2]==1)
    mask_back = (edges[:,0,2]==0)&(edges[:,1,2]==0)
    ax.add_collection3d(Line3DCollection(edges[mask_top], colors='k', linestyles='dashed', linewidths=2))
    ax.add_collection3d(Line3DCollection(edges[mask_back], colors='k', linestyles='dashed', linewidths=2, zorder=100))
    ax.add_collection3d(Line3DCollection(edges[~(mask_top|mask_back)], colors='k', linestyles='solid', linewidths=2))

    dashed_manual = [
        [(0,0,0),(0,1,0)], [(0,1,1),(0,1,0)], [(0,0,0),(0,0,1)],
        [(0,0,0),(1,0,0)], [(1,0,0),(1,0,1)]
    ]
    ax.add_collection3d(Line3DCollection(dashed_manual, colors='k', linestyles='dashed', linewidths=2, zorder=100))

    ax.view_init(elev=30, azim=220)
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # --- Overlay arrows + sine curves (in axes coords) ---
    overlay = fig.add_axes(ax.get_position(), frameon=False)
    overlay.set_xlim(-1, 1)
    overlay.set_ylim(-1, 1)
    overlay.set_xticks([]); overlay.set_yticks([])
    overlay.set_aspect('equal')

    origin = (.76, 0.35)
    targets = [
        (1.36, 0.68), # V 21
        (0.085, 0.62),  # V 12
        (0.76, -.45),  # V 01
    ]


    for target in targets:
        arrow = FancyArrowPatch(origin, target, transform=overlay.transAxes,
                                arrowstyle='-|>', mutation_scale=70,
                                lw=0, color=V_col, zorder=1000)
        arrow.set_clip_on(False)
        overlay.add_patch(arrow)

        vec2 = np.array(target) - np.array(origin)
        L2 = np.linalg.norm(vec2)
        u2 = vec2 / L2
        perp2 = np.array([-u2[1], u2[0]])
        t2 = np.linspace(0, .85 * L2, 200)

        for s in [1,0]:
            phase_shift = (np.random.rand() + .3) * s * np.pi
            w2 = .03 * np.sin( 1.5 * np.pi * (t2 / L2) + phase_shift)
            xs = u2[0] * t2 + perp2[0] * w2 + origin[0]
            ys = u2[1] * t2 + perp2[1] * w2 + origin[1]

            line = Line2D(xs, ys, transform=overlay.transAxes, color=V_col, linewidth=2.5, zorder=10, alpha = 1)
            line.set_clip_on(False)
            fig.add_artist(line)


        square = Line2D([origin[0]], [origin[1]],
                        marker='s', color='k', ms=15,
                        transform=overlay.transAxes,
                        zorder=2000)
        square.set_clip_on(False)
        fig.add_artist(square)

    # --- Labels in 3D ---
    labels = [
        ((.25, .1, 1.22), r'V$_{2,1}$'), ((-.22, .4, 1.22), r'V$_{1,2}$'),
        ((.82, .28, .52), r'V$_{2,0}$'), ((.2, 1.02, .52), r'V$_{1,0}$'),
        ((.5, .25, .02), r'V$_{0,2}$'), ((.2, .65, .02), r'V$_{0,1}$')
    ]
    for pos, label in labels:
        ax.text(*pos, label, fontsize=12, fontweight='bold', color=V_col, zorder=999)

if __name__ == "__main__":
    figs, allK, allQ = make_embeddings(version=1)
    V_col = 'purple' # "#852B8D" 
    for i, axs in enumerate(figs):
        if i == 0:
            Q = allQ[i][0]
            K = allK[i][0]
            angle = np.rad2deg(np.arctan2(Q[1], Q[0]) - np.arctan2(K[1], K[0])) % 360
            angle = min(angle, 360-angle)
            rvec = plot_circle(axs[0, 3], V_col, angle=angle)
            # axs[0, 3].set_title("Expansion\n" + r"Z$_{cc'} \to$ V$_{cc'}$", fontsize=20, fontweight='bold', color=V_col)
            axs[0, 3].set_title("Expansion", fontsize=20, fontweight='bold', color=V_col)
            plot_polygon_expansion(axs[1, 3], V_col)

            fig = plt.gcf()
            fig.delaxes(axs[2, 3])  # remove the placeholder 2D axis
            axs[2, 3] = fig.add_subplot(3, 5, 14, projection='3d')  # row=2, col=3 -> pos 3*5 + 3 = 15
            plot_cube_expansion(axs[2, 3], fig, V_col)

        fill_manifolds(axs, V_col, rvec)
    plt.show()
