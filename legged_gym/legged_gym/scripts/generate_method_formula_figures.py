"""Render chapter-three method equations as publication-ready PNG panels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PANELS = [
    (
        "01_depth_to_bev.png",
        "Depth Back-Projection and BEV Obstacle Mapping",
        [
            r"$f_x=\frac{W}{2\tan(\phi_h/2)},\quad \phi_v=2\arctan\!\left(\tan(\phi_h/2)\frac{H}{W}\right),\quad f_y=\frac{H}{2\tan(\phi_v/2)}$",
            r"$\delta_u=\frac{u-c_x}{f_x},\quad \delta_v=\frac{v-c_y}{f_y}$",
            r"$x=d(\cos p-\delta_v\sin p),\quad y=s_l\delta_u x,\quad z=h_c-d(\sin p+\delta_v\cos p)$",
            r"$O_{ij}=\mathbb{I}\!\left[n_{ij}\geq n_{\min}\ \wedge\ (z^{\max}_{ij}-z^{\min}_{ij})>h_{\mathrm{th}}\right]$",
        ],
        "W,H: image size; fx,fy: focal lengths; u,v: pixel; d: axial depth; p: pitch; hc: camera height; Oij: occupancy.",
    ),
    (
        "02_fmm_cost_layers.png",
        "Fixed-Inflation FMM and Dual-Cost FMM",
        [
            r"$d(\mathbf{q})=\min_{\mathbf{o}\in\mathcal{O}}\|\mathbf{q}-\mathbf{o}\|_2,\qquad \mathcal{O}_h=\mathcal{O}\oplus\mathcal{B}(r_h)$",
            r"$C_s(\mathbf{q})=\left[\mathrm{clip}\!\left(\frac{r_s-d(\mathbf{q})}{r_s-r_h},0,1\right)\right]^p$",
            r"$F(\mathbf{q})=\frac{1}{1+w_sC_s(\mathbf{q})},\qquad F(\mathbf{q})=0\ \ (\mathbf{q}\in\mathcal{O}_h)$",
            r"$\|\nabla T(\mathbf{q})\|_2F(\mathbf{q})=1,\qquad T(\mathbf{q}_g)=0$",
            r"$\mathrm{Fixed\ FMM:}\quad\mathcal{O}_{fix}=\mathcal{O}\oplus\mathcal{B}(r_{fix}),\quad \|\nabla D\|_2=1$",
        ],
        "q: grid cell; O: raw obstacles; rh: hard radius; rs: soft safety distance; Cs: soft cost; F: propagation speed; T: arrival-time field.",
    ),
    (
        "03_footprint_motion_model.png",
        "Body Footprint and Short-Horizon Motion Prediction",
        [
            r"$\mathbf{x}_t=[x_t,y_t,\theta_t]^\mathsf{T},\qquad \mathbf{u}_t=[v_t,\omega_t]^\mathsf{T}$",
            r"$x_{t+1}=x_t+v_t\cos\theta_t\Delta t,\quad y_{t+1}=y_t+v_t\sin\theta_t\Delta t$",
            r"$\theta_{t+1}=\mathrm{wrap}(\theta_t+\omega_t\Delta t)$",
            r"$\mathcal{F}(\mathbf{x}_t)=\mathbf{p}_t+R(\theta_t)\mathcal{F}_0,\quad I_t=\mathbb{I}[\mathcal{F}(\mathbf{x}_t)\cap\mathcal{O}\ne\varnothing]$",
            r"$L_f=L+2m_s,\qquad W_f=W_b+2m_s$",
        ],
        "x,y,theta: predicted pose; v,omega: linear and angular velocity; F0: rectangular footprint; R: rotation matrix; ms: safety margin.",
    ),
    (
        "04_mppi_trajectory_cost.png",
        "Footprint MPPI Candidate-Trajectory Cost",
        [
            r"$J_k=J_{trk}+J_{ter}+J_{head}+J_{col}+J_{clr}+J_{sm}+J_{ctl}$",
            r"$J_{trk}=5\sum_{t=1}^{T}\|\mathbf{p}_t^{(k)}-\mathbf{r}_t\|_2^2,\quad J_{ter}=15\|\mathbf{p}_T^{(k)}-\mathbf{r}_T\|_2^2$",
            r"$J_{head}=3\,\mathrm{wrap}(\theta_T^*-\theta_T^{(k)})^2,\quad J_{col}=2500\sum_{t=1}^{T}I_t^{(k)}$",
            r"$J_{clr}=18\sum_{t=1}^{T}\left[\frac{\max(0,0.65-d_t^{(k)})}{0.65}\right]^2$",
            r"$J_{sm}=1.5\sum_{t=2}^{T}(\Delta v_t)^2+4\sum_{t=2}^{T}(\Delta\omega_t)^2$",
            r"$J_{ctl}=0.08\sum_{t=1}^{T}(v_{max}-v_t)^2+0.35\sum_{t=1}^{T}\omega_t^2$",
        ],
        "k: sample index; r: FMM reference; d: obstacle safety margin; I: footprint-collision indicator; T: prediction horizon; numbers are current weights.",
    ),
    (
        "05_mppi_sampling_update.png",
        "MPPI Sampling, Weighted Update, and Control Smoothing",
        [
            r"$\mathbf{U}^{(k)}=\mathrm{clip}(\bar{\mathbf{U}}+\vec{\epsilon}^{(k)}),\qquad k=1,\ldots,K$",
            r"$\vec{\epsilon}_t^{(k)}=0.65\vec{\epsilon}_{t-1}^{(k)}+0.35\vec{\eta}_t^{(k)}$",
            r"$\rho=\min_kJ_k,\qquad w_k=\frac{\exp[-(J_k-\rho)/\lambda]}{\sum_{j=1}^{K}\exp[-(J_j-\rho)/\lambda]}$",
            r"$\bar{\mathbf{U}}\leftarrow\sum_{k=1}^{K}w_k\mathbf{U}^{(k)},\qquad \mathbf{u}_{cmd}=\bar{\mathbf{u}}_0$",
            r"$\omega_{cmd}\leftarrow0.65\omega_{prev}+0.35\omega_{cmd},\quad |\Delta v|\leq0.04,\quad |\Delta\omega|\leq0.08$",
        ],
        "U: control sequence; epsilon/eta: correlated perturbation and Gaussian noise; K=384; lambda=8; execute only the first control.",
    ),
    (
        "06_comparison_planners.png",
        "Core Expressions of SAPF, PS-DWA, and iPlanner",
        [
            r"$\mathbf{F}_{att}=k_a\frac{\mathbf{q}_g-\mathbf{q}}{\|\mathbf{q}_g-\mathbf{q}\|_2},\quad \mathbf{F}_{SAPF}=\mathbf{F}_{att}+R(\gamma)\mathbf{F}_{rep}$",
            r"$\|\mathbf{F}_{rep}\|=k_r\left(\frac{1}{d-r_{safe}}-\frac{1}{r_{inf}-r_{safe}}\right)\frac{1}{(d-r_{safe})^2}$",
            r"$\mathcal{V}_t=[v_t-a_v\Delta t,v_t+a_v\Delta t]\times[\omega_t-a_\omega\Delta t,\omega_t+a_\omega\Delta t]$",
            r"$S(v,\omega)=4S_d+2S_h+0.5S_v,\qquad (v^*,\omega^*)=\arg\max_{(v,\omega)\in\mathcal{V}_t}S$",
            r"$(\hat{\mathcal{P}},\hat f)=g_{\theta}(D,\mathbf{g}),\qquad \mathbf{u}=\mathcal{T}(\hat{\mathcal{P}},\hat f)$",
        ],
        "SAPF rotates repulsion for obstacle flow; PS-DWA optimizes the dynamic window by pattern search; iPlanner predicts path P and fear value f from depth D and goal g.",
    ),
    (
        "07_himloco_ppo.png",
        "HIMLoco Policy and PPO Objective",
        [
            r"$\mathbf{a}_t=\pi_{\theta}(\mathbf{o}_t,\mathbf{h}_t,\hat{\mathbf{z}}_t),\qquad \mathbf{q}_t^{des}=\mathbf{q}^{def}+s_a\mathbf{a}_t$",
            r"$r_t(\theta)=\frac{\pi_{\theta}(\mathbf{a}_t|\mathbf{o}_t)}{\pi_{\theta_{old}}(\mathbf{a}_t|\mathbf{o}_t)}$",
            r"$L_{clip}=\mathbb{E}_t\!\left[\min\!\left(r_t\hat A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]$",
            r"$L_V=\mathbb{E}_t[(V_{\psi}(\mathbf{o}_t)-\hat R_t)^2]$",
            r"$L=-L_{clip}+c_vL_V-c_e\mathcal{H}(\pi_{\theta})+L_{est}+L_{swap}$",
        ],
        "o,h: current and historical observations; z: latent variable; a: 12-D joint action; A: advantage; V: value function; H: policy entropy; epsilon: clipping threshold.",
    ),
    (
        "08_evaluation_metrics.png",
        "Navigation and Locomotion Metrics",
        [
            r"$e_v=\frac{1}{N}\sum_{t=1}^{N}|v_t^{cmd}-v_t^{real}|$",
            r"$e_{ori}=\frac{1}{N}\sum_{t=1}^{N}\sqrt{\phi_t^2+\vartheta_t^2},\qquad e_{ang}=\frac{1}{N}\sum_{t=1}^{N}\sqrt{\omega_{x,t}^2+\omega_{y,t}^2}$",
            r"$S_{\omega}=\frac{1}{N-1}\sum_{t=2}^{N}|\omega_t^{cmd}-\omega_{t-1}^{cmd}|$",
            r"$E_{\tau}=\frac{1}{N}\sum_{t=1}^{N}\sum_{j=1}^{12}\tau_{j,t}^{2},\qquad \bar t_p=\frac{1}{N_p}\sum_{i=1}^{N_p}t_{p,i}$",
            r"$\mathrm{SuccessRate}=\frac{N_{success}}{N_{trial}}\times100\%$",
        ],
        "N: control steps; phi,vartheta: roll and pitch; tau: joint torque; tp: planning time; Ntrial: number of trials.",
    ),
    (
        "09_umppi_unscented_risk.png",
        "U-MPPI Unscented Sampling and Risk-Sensitive Cost",
        [
            r"$\lambda_{\sigma}=\alpha^2(n_x+\kappa)-n_x,\qquad n_{\sigma}=2n_x+1$",
            r"$\mathcal{X}_0=\bar{\mathbf{x}},\quad \mathcal{X}_i=\bar{\mathbf{x}}+[\sqrt{(n_x+\lambda_{\sigma})\Sigma}]_i,\quad \mathcal{X}_{i+n_x}=\bar{\mathbf{x}}-[\sqrt{(n_x+\lambda_{\sigma})\Sigma}]_i$",
            r"$\mathcal{X}_{t+1}^{(i)}=f(\mathcal{X}_t^{(i)},\mathbf{u}_t+\delta\mathbf{u}_{t,m}),\quad \bar{\mathbf{x}}_{t+1}=\sum_iw_i^{(m)}\mathcal{X}_{t+1}^{(i)}$",
            r"$\Sigma_{t+1}=\sum_iw_i^{(c)}(\mathcal{X}_{t+1}^{(i)}-\bar{\mathbf{x}}_{t+1})(\mathcal{X}_{t+1}^{(i)}-\bar{\mathbf{x}}_{t+1})^{\mathsf{T}}$",
            r"$q_{rs}=\frac{1}{\gamma}\log\det(\mathbf{I}+\gamma\mathbf{Q}\Sigma)+\|\bar{\mathbf{x}}-\mathbf{x}_{ref}\|_{(\mathbf{Q}^{-1}+\gamma\Sigma)^{-1}}^2$",
            r"$\bar{\mathbf{U}}\leftarrow\mathrm{SGF}\!\left(\sum_m\frac{\exp[-(J_m-\rho)/\lambda]}{\sum_j\exp[-(J_j-\rho)/\lambda]}\mathbf{U}^{(m)}\right)$",
        ],
        "nx=3: state dimension; X: sigma point; Sigma: state covariance; gamma: risk parameter; Q: tracking weight; SGF: Savitzky-Golay filter.",
    ),
]


def configure_font():
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


def render_panel(output, title, equations, note):
    height = 1.25 + 0.72 * len(equations) + 0.75
    fig = plt.figure(figsize=(12.0, height), facecolor="white")
    ax = fig.add_axes([0.025, 0.04, 0.95, 0.92])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.004, 0.004), 0.992, 0.992,
                               facecolor="#FFFFFF", edgecolor="#64748B", linewidth=1.2))
    ax.text(0.035, 0.92, title, fontsize=20, fontweight="bold",
            color="#111827", va="center")
    start_y = 0.79
    step = 0.60 / max(1, len(equations) - 1) if len(equations) > 1 else 0
    for index, equation in enumerate(equations):
        ax.text(0.055, start_y - index * step, equation, fontsize=17,
                color="#0F172A", va="center")
    ax.add_patch(plt.Rectangle((0.03, 0.035), 0.94, 0.105,
                               facecolor="#F1F5F9", edgecolor="none"))
    ax.text(0.05, 0.087, note, fontsize=10.5, color="#334155",
            va="center", wrap=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_font()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, title, equations, note in PANELS:
        render_panel(args.output / filename, title, equations, note)
    print("Generated %d formula PNGs in %s" % (len(PANELS), args.output))


if __name__ == "__main__":
    main()
