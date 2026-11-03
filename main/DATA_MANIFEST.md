# main/DATA manifest

22,667 files, 211.9 GB. Tiers: archive (checkpoints, run logs, the probe set: `python main/data_tools.py fetch`), cache (regenerate: see the modules below and their docstrings), light (in git under main/DATA_light: `python main/data_tools.py light-restore`). Per-file list: DATA_MANIFEST.tsv.

| folder | size | files | archive | light | mentioned by |
|---|---|---|---|---|---|
| RL_state_num_reps | 0.64 GB | 6 | 0.64 GB | 0.0 MB | compare_RL_state_num.py |
| cache | 17.39 GB | 10 | 0.00 GB | 0.0 MB | compare_matched_params.py, compare_sanity_reps.py, fig1c_scan_hidden.py, gradflow_analysis.py, plot_ablation.py, plot_fig1c_all.py ... |
| controller | 0.02 GB | 3 | 0.00 GB | 0.0 MB | plot_controllers.py |
| experimental | 174.00 GB | 22,552 | 86.31 GB | 111.3 MB | fig1.py, fig1_movie.py, fig1c_joint_entanglement.py, fig1c_jump_dims.py, fig1c_lesion_test.py, fig1c_mechanisms.py ... |
| experimental/esn_learnability | 0.37 GB | 1,666 | 0.31 GB | 12.9 MB | fig_minimal_set.py, plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/esn_learnability/full_esn_entropy | 0.31 GB | 250 | 0.31 GB | 0.0 MB | plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/esn_learnability/readout_only | 0.01 GB | 15 | 0.00 GB | 11.2 MB | plot_episode_start_curves.py |
| experimental/esn_learnability/theory | 0.04 GB | 1,115 | 0.00 GB | 0.4 MB | fig_minimal_set.py |
| experimental/esn_mlp_rescue | 0.07 GB | 4 | 0.07 GB | 0.0 MB | plot_ff_deep.py |
| experimental/esn_widths | 0.14 GB | 2 | 0.13 GB | 0.0 MB | plot_matched_params_new.py |
| experimental/ff_deep | 30.92 GB | 3,795 | 29.79 GB | 3.6 MB | plot_ff_deep.py, plot_naive_vs_linear.py, td_questions.py, td_questions_frozen.py |
| experimental/ff_deep/_continue_check | 0.07 GB | 57 | 0.07 GB | 0.0 MB |  |
| experimental/ff_deep/d4_variants | 0.29 GB | 38 | 0.27 GB | 0.0 MB | plot_ff_deep.py |
| experimental/ff_deep/d9_frozen | 0.57 GB | 60 | 0.53 GB | 0.0 MB | plot_ff_deep.py, td_questions.py, td_questions_frozen.py |
| experimental/ff_deep/extrapolation | 0.48 GB | 2,024 | 0.00 GB | 3.5 MB | plot_ff_deep.py, plot_naive_vs_linear.py |
| experimental/ff_deep/lr_sweep | 0.01 GB | 216 | 0.00 GB | 0.0 MB |  |
| experimental/ff_deep/shared_readin_check | 0.17 GB | 346 | 0.00 GB | 0.0 MB |  |
| experimental/ff_deep/widths | 28.65 GB | 838 | 28.30 GB | 0.0 MB | plot_ff_deep.py, td_tt_memlen.py |
| experimental/fig1 | 1.42 GB | 656 | 0.00 GB | 0.2 MB | fig1.py, fig1_checks.py, fig1_movie.py, fig1c_crossings.py, fig1c_entanglement_definitions.py, fig1c_figure8_learning.py ... |
| experimental/fig1c | 2.15 GB | 19 | 0.00 GB | 4.5 MB | fig1_movie.py, fig1c_crossings.py, fig1c_entanglement_definitions.py, fig1c_figure8_learning.py, fig1c_figure8_movie.py, fig1c_joint_entanglement.py ... |
| experimental/fig1c_compression | 13.90 GB | 1,708 | 0.00 GB | 21.9 MB | fig1_movie.py, fig1c_joint_entanglement.py, fig1c_jump_dims.py, fig1c_lesion_test.py, fig1c_mechanisms.py, fig1c_order_check.py ... |
| experimental/fig1c_compression/figdata | 4.85 GB | 197 | 0.00 GB | 21.9 MB | fig1c_joint_entanglement.py, fig1c_jump_dims.py, fig1c_lesion_test.py, fig1c_mechanisms.py, fig1c_order_check.py, fig1c_scan_credit.py ... |
| experimental/gradflow_ctx | 4.22 GB | 18 | 0.00 GB | 0.0 MB | fig1.py, fig1c_two_processes.py, fig_minimal_set.py, gradflow_analysis.py |
| experimental/gradflow_explain | 5.24 GB | 8 | 0.00 GB | 0.0 MB | gradflow_explain.py |
| experimental/gradflow_obs | 1.74 GB | 2 | 0.00 GB | 0.0 MB | gradflow_analysis.py |
| experimental/interaction_bounds | 0.20 GB | 793 | 0.17 GB | 0.0 MB | plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/interaction_bounds/analyses | 0.20 GB | 778 | 0.17 GB | 0.0 MB | plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/training_dynamics | 113.65 GB | 13,881 | 55.84 GB | 68.2 MB | fig_entanglement_driver.py, fig_td_summary.py, plot_ff_deep.py, plot_joint_esn_training.py, plot_matched_params_new.py, plot_training_dynamics.py ... |
| experimental/training_dynamics/analysis | 42.77 GB | 1,335 | 0.33 GB | 3.7 MB | fig_entanglement_driver.py, plot_training_dynamics.py, td_analysis.py, td_entanglement_driver.py, td_entanglement_explained.py, td_entanglement_factors.py ... |
| experimental/training_dynamics/echo_s20261029 | 4.53 GB | 358 | 4.53 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/joint_echo_jl_s20261029 | 5.92 GB | 454 | 5.92 GB | 0.0 MB | plot_ff_deep.py, plot_joint_esn_training.py, td_questions_data.py |
| experimental/training_dynamics/joint_echo_s20261029 | 5.92 GB | 454 | 5.92 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/joint_trained_s20261029 | 6.60 GB | 454 | 6.60 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/mlp55_s20261029 | 9.07 GB | 358 | 9.07 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/summary_data | 0.85 GB | 17 | 0.00 GB | 0.3 MB | fig_td_summary.py, td_grids_fwd.py, td_summary_data.py, td_tt_latent.py, td_tt_splitting.py, td_tt_wormholes.py |
| experimental/training_dynamics/through_training | 14.41 GB | 6,554 | 0.00 GB | 64.1 MB | td_tt_chaos_ent.py, td_tt_credit_lags.py, td_tt_entangled_learning.py, td_tt_flips.py, td_tt_gap.py, td_tt_intervene.py ... |
| experimental/training_dynamics/trained_N100_s20261029 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261030 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261031 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261032 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261033 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N200_s20261029 | 0.77 GB | 358 | 0.77 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_N50_s20261029 | 0.54 GB | 358 | 0.54 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_lastT_p1_s20261029 | 3.45 GB | 265 | 3.45 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_lastT_s20261029 | 5.26 GB | 358 | 5.26 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_p1_s20261029 | 5.26 GB | 358 | 5.26 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_s20261029 | 5.21 GB | 358 | 5.21 GB | 0.0 MB | td_questions_data.py |
| lazyrich | 7.15 GB | 17 | 7.11 GB | 0.0 MB | plot_lazyrich.py |
| oracle_clf_standard | 0.01 GB | 8 | 0.00 GB | 0.0 MB |  |
| oracle_pilot | 1.17 GB | 12 | 1.17 GB | 0.0 MB | plot_ablation.py, plot_oracle_losses.py |
| params | 0.86 GB | 18 | 0.86 GB | 0.0 MB | plot_ff_deep.py |
| sanity | 6.58 GB | 14 | 6.58 GB | 0.0 MB | Collector.py, compare_matched_params.py, compare_sanity_reps.py, gradflow_analysis.py, plot_episode_start_curves.py, plot_ff_deep.py ... |
| sanity_default_reps | 3.96 GB | 26 | 3.96 GB | 0.0 MB | compare_matched_params.py, compare_sanity_reps.py |
