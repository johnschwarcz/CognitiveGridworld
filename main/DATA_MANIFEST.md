# main/DATA manifest

37,231 files, 427.0 GB. Tiers: archive (checkpoints, run logs, the probe set: `python main/data_tools.py fetch`), cache (regenerate: see the modules below and their docstrings), light (in git under main/DATA_light: `python main/data_tools.py light-restore`). Per-file list: DATA_MANIFEST.tsv.

| folder | size | files | archive | light | mentioned by |
|---|---|---|---|---|---|
| RL_state_num_reps | 0.64 GB | 6 | 0.64 GB | 0.0 MB | bayes_gen_fr_div.py, compare_RL_state_num.py |
| cache | 17.41 GB | 12 | 0.00 GB | 2.3 MB | compare_matched_params.py, compare_sanity_reps.py, fig13_paper_form.py, fig13_plugin_lead.py, fig13_rcc_arm.py, fig13_shuffled_routes.py ... |
| cluster_runs | 7.29 GB | 71 | 7.29 GB | 0.0 MB | bayes_gen_fr_div.py, bayes_gen_plot.py, cluster_runs_plot.py, fig12_default_rcc.py, fig13_embeddings.py, fig13_memorization.py ... |
| controller | 0.02 GB | 3 | 0.00 GB | 0.0 MB | plot_controllers.py |
| experimental | 375.98 GB | 36,962 | 277.30 GB | 162.7 MB | bayes_gen_arch_plot.py, bayes_gen_controls_plot.py, bayes_gen_fr_div.py, bayes_gen_plot.py, cluster_runs_plot.py, credit_time_control.py ... |
| experimental/bayes_gen_fr | 4.08 GB | 162 | 3.89 GB | 0.0 MB | bayes_gen_fr_div.py |
| experimental/bayes_gen_fr/nets | 3.89 GB | 54 | 3.89 GB | 0.0 MB | bayes_gen_fr_div.py, fig13_embeddings.py, fig_channel_swap.py, fig_routes_status.py |
| experimental/cluster_runs | 0.61 GB | 283 | 0.41 GB | 0.1 MB | bayes_gen_fr_div.py, bayes_gen_plot.py, cluster_runs_plot.py, fig12_default_rcc.py, fig13_embeddings.py, fig13_memorization.py ... |
| experimental/cluster_runs/jsonl | 0.20 GB | 276 | 0.00 GB | 0.0 MB | bayes_gen_controls_plot.py, bayes_gen_plot.py, cluster_runs_plot.py, fig12_default_rcc.py, fig13_memorization.py, fig13_paper_form.py ... |
| experimental/esn_convex | 0.02 GB | 60 | 0.00 GB | 0.0 MB | esn_convex_plot.py |
| experimental/esn_learnability | 0.35 GB | 1,493 | 0.31 GB | 13.5 MB | fig_minimal_set.py, learnability_checks.py, plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/esn_learnability/full_esn_entropy | 0.31 GB | 250 | 0.31 GB | 0.0 MB | plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/esn_learnability/readout_only | 0.01 GB | 15 | 0.00 GB | 11.2 MB | plot_episode_start_curves.py |
| experimental/esn_learnability/theory | 0.02 GB | 933 | 0.00 GB | 0.4 MB | fig_channel_loss.py, fig_minimal_set.py, fr_theory.py, fr_theory_figure.py |
| experimental/esn_mlp_rescue | 0.07 GB | 4 | 0.07 GB | 0.0 MB | plot_ff_deep.py |
| experimental/esn_widths | 0.14 GB | 2 | 0.13 GB | 0.0 MB | learnability_checks.py, plot_matched_params_new.py |
| experimental/ff_deep | 30.92 GB | 3,798 | 29.79 GB | 3.6 MB | plot_ff_deep.py, plot_log_trend.py, plot_naive_vs_linear.py, td_questions.py, td_questions_frozen.py |
| experimental/ff_deep/_continue_check | 0.07 GB | 57 | 0.07 GB | 0.0 MB |  |
| experimental/ff_deep/d4_variants | 0.29 GB | 38 | 0.27 GB | 0.0 MB | plot_ff_deep.py |
| experimental/ff_deep/d9_frozen | 0.57 GB | 60 | 0.53 GB | 0.0 MB | plot_ff_deep.py, td_questions.py, td_questions_frozen.py |
| experimental/ff_deep/extrapolation | 0.48 GB | 2,027 | 0.00 GB | 3.5 MB | plot_ff_deep.py, plot_log_trend.py, plot_naive_vs_linear.py |
| experimental/ff_deep/lr_sweep | 0.01 GB | 216 | 0.00 GB | 0.0 MB |  |
| experimental/ff_deep/shared_readin_check | 0.17 GB | 346 | 0.00 GB | 0.0 MB |  |
| experimental/ff_deep/widths | 28.65 GB | 838 | 28.30 GB | 0.0 MB | plot_ff_deep.py, td_tt_memlen.py |
| experimental/fig1 | 1.42 GB | 653 | 0.00 GB | 0.2 MB | fig1.py, fig12_default_rcc.py, fig13_embeddings.py, fig1_checks.py, fig1_movie.py, fig1c_crossings.py ... |
| experimental/fig12_default_rcc | 0.10 GB | 1 | 0.10 GB | 0.0 MB | fig12_default_rcc.py |
| experimental/fig13_embeddings | 2.60 GB | 251 | 2.58 GB | 0.8 MB | fig13_controls.py, fig13_embeddings.py, fig13_fresh_generator.py, fig13_summary.py, fig13_transfer.py, fig_generator_route.py ... |
| experimental/fig13_embeddings/controls | 0.01 GB | 36 | 0.00 GB | 0.0 MB | fig13_controls.py, fig1c_jump_dims.py, fig_generator_route.py, fig_matched_learners.py, learnability_checks.py, td_tt_memcuts.py |
| experimental/fig13_embeddings/nets | 2.58 GB | 26 | 2.58 GB | 0.0 MB | bayes_gen_fr_div.py, fig13_embeddings.py, fig_channel_swap.py, fig_routes_status.py |
| experimental/fig1c | 2.36 GB | 25 | 0.00 GB | 4.5 MB | fig1_movie.py, fig1c_crossings.py, fig1c_entanglement_definitions.py, fig1c_figure8_learning.py, fig1c_figure8_movie.py, fig1c_joint_entanglement.py ... |
| experimental/fig1c_compression | 14.32 GB | 1,725 | 0.00 GB | 21.9 MB | fig1_movie.py, fig1c_joint_entanglement.py, fig1c_jump_dims.py, fig1c_lesion_test.py, fig1c_mechanisms.py, fig1c_order_check.py ... |
| experimental/fig1c_compression/figdata | 5.25 GB | 212 | 0.00 GB | 21.9 MB | fig1c_joint_entanglement.py, fig1c_jump_dims.py, fig1c_lesion_test.py, fig1c_mechanisms.py, fig1c_order_check.py, fig1c_scan_credit.py ... |
| experimental/fr_definition | 0.62 GB | 31 | 0.46 GB | 0.3 MB | fig2b_fr_panel.py, fig_manifold_fr.py, fig_supp_diagnostics.py, fr_correct_fresh.py, fr_definition_check.py, supplementary_plotter2.py |
| experimental/fr_definition/clamp_nets | 0.46 GB | 6 | 0.46 GB | 0.0 MB |  |
| experimental/fr_definition/fig1c_forward | 0.14 GB | 7 | 0.00 GB | 0.0 MB |  |
| experimental/fr_theory | 2.58 GB | 262 | 0.00 GB | 0.5 MB | fr_definition_check.py, fr_theory.py, fr_theory_additions.py, fr_theory_additions_plot.py, fr_theory_dipsim.py, fr_theory_figure.py ... |
| experimental/fr_theory/additions | 0.01 GB | 40 | 0.00 GB | 0.2 MB | fr_theory_additions.py, fr_theory_dipsim.py, fr_theory_horizon.py |
| experimental/fr_theory/baseline | 0.49 GB | 13 | 0.00 GB | 0.0 MB | explain_interaction.py, fig1.py, fig13_shuffled_routes.py, fig1_checks.py, fig1_movie.py, fig1c_mechanisms.py ... |
| experimental/fr_theory/distinct_pairs | 0.49 GB | 13 | 0.00 GB | 0.0 MB |  |
| experimental/fr_theory/fr_theory_cache | 1.57 GB | 37 | 0.00 GB | 0.0 MB | fr_definition_check.py, fr_theory.py, fr_theory_additions_plot.py, fr_theory_dipsim.py, fr_theory_figure.py, fr_theory_horizon.py |
| experimental/fr_theory/fr_theory_inspect | 0.01 GB | 60 | 0.00 GB | 0.0 MB | fr_theory.py, fr_theory_figure.py |
| experimental/gradflow_ctx | 4.22 GB | 18 | 0.00 GB | 0.0 MB | fig1.py, fig1c_two_processes.py, fig_minimal_set.py, gradflow_analysis.py |
| experimental/gradflow_explain | 5.24 GB | 8 | 0.00 GB | 0.0 MB | gradflow_explain.py |
| experimental/gradflow_obs | 1.74 GB | 2 | 0.00 GB | 0.0 MB | gradflow_analysis.py |
| experimental/interaction_bounds | 0.20 GB | 787 | 0.17 GB | 0.0 MB | plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/interaction_bounds/analyses | 0.20 GB | 777 | 0.17 GB | 0.0 MB | plot_episode_start_curves.py, plot_ff_deep.py, plot_joint_esn_training.py |
| experimental/rcc_pairs | 0.16 GB | 4 | 0.00 GB | 0.0 MB | rcc_pairs.py |
| experimental/training_dynamics | 304.23 GB | 26,572 | 239.39 GB | 117.2 MB | credit_time_control.py, fig_entanglement_driver.py, fig_learn_fr.py, fig_learn_grids_pr.py, fig_learn_interaction.py, fig_learn_lever.py ... |
| experimental/training_dynamics/analysis | 42.86 GB | 1,379 | 0.33 GB | 4.7 MB | credit_time_control.py, fig_entanglement_driver.py, fig_learn_fr.py, fig_learn_grids_pr.py, fig_learn_lever.py, fig_learn_mechanism.py ... |
| experimental/training_dynamics/echo_N10000_b8k_s20261029 | 5.17 GB | 59 | 5.17 GB | 0.0 MB |  |
| experimental/training_dynamics/echo_N200_s20261029 | 0.11 GB | 59 | 0.11 GB | 0.0 MB |  |
| experimental/training_dynamics/echo_N500_s20261029 | 1.60 GB | 358 | 1.60 GB | 0.0 MB |  |
| experimental/training_dynamics/echo_s20261029 | 4.53 GB | 358 | 4.53 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/joint_echo_N10000_b8k_s20261029 | 5.19 GB | 59 | 5.19 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N10000_jl_s20261029 | 5.19 GB | 59 | 5.19 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N10000_s20261029 | 5.19 GB | 59 | 5.19 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N2000_jl_s20261029 | 0.75 GB | 58 | 0.75 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N2000_s20261029 | 0.75 GB | 58 | 0.75 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N200_jl_s20261029 | 0.11 GB | 59 | 0.11 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N200_s20261029 | 0.11 GB | 59 | 0.11 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N500_jl_s20261029 | 2.15 GB | 454 | 2.15 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_N500_s20261029 | 0.22 GB | 58 | 0.22 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_echo_jl_s20261029 | 5.92 GB | 454 | 5.92 GB | 0.0 MB | plot_ff_deep.py, plot_joint_esn_training.py, td_questions_data.py |
| experimental/training_dynamics/joint_echo_s20261029 | 5.92 GB | 454 | 5.92 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/joint_trained_N100_jl_s20261029 | 0.88 GB | 454 | 0.88 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N100_s20261029 | 0.88 GB | 454 | 0.88 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N150_jl_s20261029 | 0.09 GB | 59 | 0.09 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N150_s20261029 | 0.09 GB | 59 | 0.09 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N200_jl_s20261029 | 1.09 GB | 454 | 1.09 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N200_s20261029 | 1.09 GB | 454 | 1.09 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N300_jl_s20261029 | 0.16 GB | 59 | 0.16 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N300_s20261029 | 0.16 GB | 59 | 0.16 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N400_jl_s20261029 | 0.20 GB | 59 | 0.20 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N400_s20261029 | 0.20 GB | 59 | 0.20 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N500_jl_s20261029 | 0.26 GB | 59 | 0.26 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N500_s20261029 | 0.26 GB | 59 | 0.26 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N50_jl_s20261029 | 0.81 GB | 454 | 0.81 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_N50_s20261029 | 0.81 GB | 454 | 0.81 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_jl_s20261029 | 6.60 GB | 454 | 6.60 GB | 0.0 MB |  |
| experimental/training_dynamics/joint_trained_s20261029 | 6.60 GB | 454 | 6.60 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/learnmask_ctrl_s0_s20261029 | 1.74 GB | 51 | 1.74 GB | 0.2 MB |  |
| experimental/training_dynamics/learnmask_ctrl_s20261029 | 0.39 GB | 14 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_hi_f05_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_hi_f50_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_hi_f50_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_hi_fr_f50_s0_r2_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_hi_fr_f50_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_hi_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_lo_f05_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_lo_f50_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_lo_f50_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_lo_fr_f50_s0_r2_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_lo_fr_f50_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_lo_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_rand_f05_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_rand_f50_s0_r2_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_rand_f50_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_det_rand_f50_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_det_rand_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_excl_bandhi_b01_1000_s0_s20261029_short_killed | 1.11 GB | 35 | 1.11 GB | 0.4 MB |  |
| experimental/training_dynamics/learnmask_excl_bandhi_b01_1000_s0_s20261029_stuck_killed | 2.34 GB | 70 | 2.34 GB | 1.5 MB |  |
| experimental/training_dynamics/learnmask_excl_bandhi_b03_300_c03_300_s0_s20261029 | 24.59 GB | 700 | 24.59 GB | 2.0 MB |  |
| experimental/training_dynamics/learnmask_excl_bandhi_b10_100_c10_100_s0_s20261029 | 14.20 GB | 408 | 14.20 GB | 5.6 MB |  |
| experimental/training_dynamics/learnmask_excl_bandlo_b01_1000_s0_s20261029_short_killed | 1.25 GB | 39 | 1.25 GB | 0.5 MB |  |
| experimental/training_dynamics/learnmask_excl_bandlo_b01_1000_s0_s20261029_stuck_killed | 2.62 GB | 78 | 2.62 GB | 1.7 MB |  |
| experimental/training_dynamics/learnmask_excl_bandlo_b03_300_c03_300_s0_s20261029 | 24.59 GB | 700 | 24.59 GB | 2.0 MB |  |
| experimental/training_dynamics/learnmask_excl_bandlo_b10_100_c10_100_s0_s20261029 | 14.20 GB | 408 | 14.20 GB | 5.6 MB |  |
| experimental/training_dynamics/learnmask_excl_bandrand_b10_100_c10_100_s0_s20261029 | 14.20 GB | 408 | 14.20 GB | 5.6 MB |  |
| experimental/training_dynamics/learnmask_excl_hi_thr01_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_excl_hi_thr100_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_excl_hi_thr10_s0_s20261029 | 4.89 GB | 142 | 4.89 GB | 3.6 MB |  |
| experimental/training_dynamics/learnmask_excl_hi_thr300_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_excl_hi_thr30_s0_s20261029 | 0.90 GB | 29 | 0.90 GB | 0.2 MB |  |
| experimental/training_dynamics/learnmask_excl_lo_thr01_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_excl_lo_thr100_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_excl_lo_thr10_s0_s20261029 | 4.89 GB | 142 | 4.89 GB | 3.6 MB |  |
| experimental/training_dynamics/learnmask_excl_lo_thr300_s0_s20261029 | 0.97 GB | 31 | 0.97 GB | 0.3 MB |  |
| experimental/training_dynamics/learnmask_excl_rand_thr10_s0_s20261029 | 1.74 GB | 52 | 1.74 GB | 0.9 MB |  |
| experimental/training_dynamics/learnmask_mask_hi_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_mask_lo_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_mask_rand_s20261029 | 0.39 GB | 15 | 0.39 GB | 0.1 MB |  |
| experimental/training_dynamics/learnmask_trunk_s20261029 | 0.83 GB | 26 | 0.83 GB | 0.0 MB |  |
| experimental/training_dynamics/mlp55_s20261029 | 9.07 GB | 358 | 9.07 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/summary_data | 1.04 GB | 21 | 0.00 GB | 0.3 MB | fig_learn_fr.py, fig_td_summary.py, td_grids_fwd.py, td_summary_data.py, td_tt_latent.py, td_tt_splitting.py ... |
| experimental/training_dynamics/through_training | 20.92 GB | 9,345 | 0.00 GB | 70.6 MB | fig_learn_fr.py, fig_learn_holdout.py, fig_learn_interaction.py, fig_learn_lever.py, fig_learn_mechanism.py, fig_learn_profile.py ... |
| experimental/training_dynamics/trained_N100_s20261029 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261030 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261031 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261032 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N100_s20261033 | 0.60 GB | 358 | 0.60 GB | 0.0 MB | td_questions.py, td_questions_data.py |
| experimental/training_dynamics/trained_N150_s20261029 | 0.09 GB | 59 | 0.09 GB | 0.0 MB |  |
| experimental/training_dynamics/trained_N200_s20261029 | 0.77 GB | 358 | 0.77 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_N300_s20261029 | 0.16 GB | 59 | 0.16 GB | 0.0 MB |  |
| experimental/training_dynamics/trained_N400_s20261029 | 0.20 GB | 59 | 0.20 GB | 0.0 MB |  |
| experimental/training_dynamics/trained_N500_s20261029 | 0.26 GB | 59 | 0.26 GB | 0.0 MB |  |
| experimental/training_dynamics/trained_N50_s20261029 | 0.54 GB | 358 | 0.54 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_lastT_p1_s20261029 | 3.45 GB | 265 | 3.45 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_lastT_s20261029 | 5.26 GB | 358 | 5.26 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_p1_s20261029 | 5.26 GB | 358 | 5.26 GB | 0.0 MB | td_questions_data.py |
| experimental/training_dynamics/trained_s20261029 | 5.21 GB | 358 | 5.21 GB | 0.0 MB | fig_learning_dynamics_forward.py, td_questions_data.py |
| lazyrich | 7.15 GB | 17 | 7.11 GB | 0.0 MB | plot_lazyrich.py |
| oracle_clf_standard | 0.01 GB | 8 | 0.00 GB | 0.0 MB |  |
| oracle_pilot | 1.17 GB | 12 | 1.17 GB | 0.0 MB | fig13_paper_form.py, fig13_rcc_arm.py, plot_ablation.py, plot_oracle_losses.py |
| params | 0.86 GB | 18 | 0.86 GB | 0.0 MB | plot_ff_deep.py |
| sanity | 6.58 GB | 14 | 6.58 GB | 0.0 MB | Collector.py, compare_matched_params.py, compare_sanity_reps.py, gradflow_analysis.py, plot_episode_start_curves.py, plot_ff_deep.py ... |
| sanity_default_reps | 3.96 GB | 26 | 3.96 GB | 0.0 MB | compare_matched_params.py, compare_sanity_reps.py |
| smoke | 5.78 GB | 81 | 5.78 GB | 0.0 MB | fig1c_lesion_test.py |
