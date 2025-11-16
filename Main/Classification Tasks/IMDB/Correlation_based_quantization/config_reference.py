"""
Configuration Presets and Templates for BERT_base_IMDB.py

This file documents standard configurations for different scenarios.
You can use these as-is or modify them for your needs.
"""

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

PRESETS = {
    "quick_test": {
        "description": "Quick 5-minute test (good for prototyping)",
        "phase_1": {
            "calibration_size": "5k (fast, noisy)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "PWCCA + CKA (recommended)"
        },
        "phase_2": {
            "eval_size": "5k (fast iteration)",
            "clustering": "K-means (recommended)",
            "n_clusters": "3 groups (simpler)",
            "bits": "[16, 8, 4] (conservative, recommended)"
        },
        "expected": {
            "phase1_time_min": 3,
            "phase2_time_min": 2,
            "accuracy_drop_pct": "2-3%",
            "compression_ratio": "2.0x"
        }
    },
    
    "balanced": {
        "description": "Balanced approach (15 minutes) - recommended for most use cases",
        "phase_1": {
            "calibration_size": "25k (full IMDB train, recommended)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "PWCCA + CKA (recommended)"
        },
        "phase_2": {
            "eval_size": "10k (medium)",
            "clustering": "K-means (recommended)",
            "n_clusters": "3 groups (simpler)",
            "bits": "[16, 8, 4] (conservative, recommended)"
        },
        "expected": {
            "phase1_time_min": 10,
            "phase2_time_min": 5,
            "accuracy_drop_pct": "1-2%",
            "compression_ratio": "2.0x"
        }
    },
    
    "conservative_final": {
        "description": "Conservative final results (30 minutes) - best accuracy, lower compression",
        "phase_1": {
            "calibration_size": "25k (full IMDB train, recommended)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "PWCCA + CKA (recommended)"
        },
        "phase_2": {
            "eval_size": "25k (full test, recommended)",
            "clustering": "K-means (recommended)",
            "n_clusters": "4 groups (finer control)",
            "bits": "[32, 16, 8, 4] (conservative, recommended)"
        },
        "expected": {
            "phase1_time_min": 10,
            "phase2_time_min": 10,
            "accuracy_drop_pct": "< 1%",
            "compression_ratio": "1.5x"
        }
    },
    
    "aggressive_compression": {
        "description": "Maximum compression (30 minutes) - focus on model size, accept accuracy loss",
        "phase_1": {
            "calibration_size": "25k (full IMDB train, recommended)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "PWCCA + CKA (recommended)"
        },
        "phase_2": {
            "eval_size": "25k (full test, recommended)",
            "clustering": "Hierarchical clustering",
            "n_clusters": "4 groups (finer control)",
            "bits": "[8, 4, 2, 2] (aggressive)"
        },
        "expected": {
            "phase1_time_min": 10,
            "phase2_time_min": 10,
            "accuracy_drop_pct": "5-10%",
            "compression_ratio": "4.0x"
        }
    },
    
    "exploratory": {
        "description": "Exploratory analysis (45 minutes) - compare multiple approaches",
        "phase_1": {
            "calibration_size": "25k (full IMDB train, recommended)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "All three (PWCCA + SVCCA + CKA)"
        },
        "phase_2": {
            "eval_size": "5k (fast iteration)",
            "clustering": "All three (K-means, Percentile, Hierarchical)",
            "n_clusters": "3 groups (simpler) [then try 4]",
            "bits": "[16, 8, 4] [then try [32, 16, 8, 4]]"
        },
        "expected": {
            "phase1_time_min": 25,
            "phase2_time_min": 20,
            "accuracy_drop_pct": "1-3% (varies by config)",
            "compression_ratio": "1.5-2.0x"
        }
    },
    
    "finetuned_bert_focus": {
        "description": "Fine-tuned BERT optimization (30 minutes)",
        "phase_1": {
            "calibration_size": "25k (full IMDB train, recommended)",
            "sampling": "Stratified sampling (recommended)",
            "metrics": "PWCCA + CKA (recommended)"
        },
        "phase_2": {
            "eval_size": "25k (full test, recommended)",
            "clustering": "Hierarchical clustering",  # better for fine-tuned models
            "n_clusters": "4 groups (finer control)",
            "bits": "[32, 16, 8, 4] (conservative, recommended)"
        },
        "notes": [
            "Fine-tuned models are more sensitive to quantization",
            "Keep classifier head and embedding in FP32",
            "Last 2 encoder layers should use 16-bit minimum",
            "If accuracy drops > 3%, consider QAT (1-3 epochs)"
        ],
        "expected": {
            "phase1_time_min": 10,
            "phase2_time_min": 10,
            "accuracy_drop_pct": "1-2%",
            "compression_ratio": "1.5x"
        }
    }
}

# ============================================================================
# STEP-BY-STEP DECISION GUIDES
# ============================================================================

DECISION_GUIDES = {
    "calibration_size": {
        "title": "How much calibration data should I use?",
        "options": {
            "5k (fast, noisy)": {
                "use_when": [
                    "Prototyping/exploring ideas",
                    "Limited time (< 5 min)",
                    "Want quick feedback"
                ],
                "pros": ["Fast (2-3 minutes)", "Good for iteration"],
                "cons": ["Noisier sensitivity estimates", "May miss important patterns"]
            },
            "10k (medium)": {
                "use_when": [
                    "Balance between speed and stability",
                    "Medium time budget (5-10 min)"
                ],
                "pros": ["Medium speed", "Better stability than 5k"],
                "cons": ["Not as stable as 25k"]
            },
            "25k (full IMDB train) [RECOMMENDED]": {
                "use_when": [
                    "Want best sensitivity estimates",
                    "Computing final results",
                    "Have time (10-15 min)"
                ],
                "pros": ["Most stable", "Best represents full training distribution"],
                "cons": ["Slower (10+ minutes)"],
                "recommendation": "Use this for final results. Use 5k for quick iteration."
            }
        }
    },
    
    "similarity_metric": {
        "title": "Which similarity metric should I use?",
        "options": {
            "PWCCA": {
                "use_when": ["Want good default", "Need speed"],
                "speed": "Fast",
                "robustness": "Good",
                "notes": "Weighted canonical correlations, standard choice"
            },
            "SVCCA": {
                "use_when": ["Want dimensionality reduction", "Many layers"],
                "speed": "Medium",
                "robustness": "Good",
                "notes": "SVD preprocessing, control via topk parameter"
            },
            "CKA": {
                "use_when": ["Want robust verification", "Finetuned models"],
                "speed": "Medium",
                "robustness": "Excellent",
                "notes": "Kernel-based, catches representational changes"
            },
            "PWCCA + CKA [RECOMMENDED]": {
                "use_when": ["Most cases", "Want verification"],
                "speed": "Medium",
                "robustness": "Excellent",
                "notes": "Best of both worlds - fast baseline + robust verification",
                "recommendation": "Use both metrics together for best results"
            },
            "All three": {
                "use_when": ["Thorough analysis", "Academic papers", "Lots of time"],
                "speed": "Slow",
                "robustness": "Excellent",
                "notes": "Compare all metrics, may find disagreements"
            }
        }
    },
    
    "clustering_method": {
        "title": "How should I cluster layers?",
        "options": {
            "K-means [RECOMMENDED]": {
                "use_when": ["Most cases", "First time", "Standard approach"],
                "speed": "Fast",
                "stability": "Stable",
                "automation": "Automatic (no tuning)",
                "notes": "Clusters layers by sensitivity values into k groups",
                "recommendation": "Default choice - works well in practice"
            },
            "Percentile bucketing": {
                "use_when": ["Want deterministic buckets", "Clear top/bottom layers"],
                "speed": "Very fast",
                "stability": "Fully deterministic",
                "automation": "No randomness",
                "notes": "Top X% → high bits, bottom X% → low bits"
            },
            "Hierarchical clustering": {
                "use_when": ["Exploratory analysis", "Fine-tuned models", "Want insights"],
                "speed": "Slow",
                "stability": "Stable",
                "automation": "Reveals natural groupings",
                "notes": "Agglomerative clustering, can dendrogram to understand structure"
            }
        }
    },
    
    "num_clusters": {
        "title": "How many precision groups should I use?",
        "options": {
            "3 groups [SIMPLER]": {
                "pros": [
                    "Simpler model",
                    "Fast to evaluate",
                    "Good compression"
                ],
                "cons": ["Less fine-grained control"],
                "bit_allocation": "[16, 8, 4] or [32, 16, 8]",
                "use_when": ["Quick iteration", "Simplicity matters"]
            },
            "4 groups [FINER CONTROL]": {
                "pros": [
                    "Better accuracy/compression tradeoff",
                    "More flexibility",
                    "Can keep heads in high precision"
                ],
                "cons": ["More complex", "Slightly slower to evaluate"],
                "bit_allocation": "[32, 16, 8, 4] or [16, 8, 4, 2]",
                "use_when": ["Final results", "Large accuracy drops with 3 groups"]
            }
        },
        "recommendation": "Start with 3 groups. If accuracy drops > 2%, use 4 groups."
    },
    
    "bit_allocation": {
        "title": "Which bit-widths should I assign to each cluster?",
        "for_3_groups": {
            "conservative": {
                "bits": "[16, 8, 4]",
                "description": "Good accuracy, moderate compression",
                "expected_drop": "1-2%",
                "compression": "2.0x",
                "recommendation": "DEFAULT - use this"
            },
            "aggressive": {
                "bits": "[8, 4, 2]",
                "description": "Max compression, higher accuracy risk",
                "expected_drop": "5-10%",
                "compression": "4.0x"
            }
        },
        "for_4_groups": {
            "conservative": {
                "bits": "[32, 16, 8, 4]",
                "description": "Excellent accuracy, moderate compression",
                "expected_drop": "< 1%",
                "compression": "1.5x",
                "recommendation": "For fine-tuned models"
            },
            "moderate": {
                "bits": "[16, 8, 4, 2]",
                "description": "Balanced accuracy and compression",
                "expected_drop": "1-2%",
                "compression": "2.5x"
            },
            "aggressive": {
                "bits": "[8, 4, 2, 2]",
                "description": "Max compression, significant accuracy loss",
                "expected_drop": "5-10%",
                "compression": "4.0x"
            }
        }
    },
    
    "eval_set_size": {
        "title": "How much test data should I use for evaluation?",
        "options": {
            "5k (fast iteration)": {
                "use_when": ["Quick feedback", "Iteration", "Limited time"],
                "pros": ["Very fast (1-2 min)", "Good for debugging"],
                "cons": ["May not be representative", "Use stratified sampling"]
            },
            "10k (medium)": {
                "use_when": ["Balance", "Confidence check"],
                "pros": ["Reasonable speed", "Better coverage"],
                "cons": []
            },
            "25k (full test) [RECOMMENDED FOR FINAL]": {
                "use_when": ["Final reported numbers", "Publication", "Benchmarks"],
                "pros": ["Full test set coverage", "Official metrics"],
                "cons": ["Slower (5+ min)"],
                "recommendation": "Use for final results, use 5k for iteration"
            }
        }
    }
}

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================

TROUBLESHOOTING = {
    "large_accuracy_drop": {
        "problem": "Accuracy dropped by 5% or more",
        "causes": [
            "Bit allocation too aggressive",
            "Calibration size too small",
            "Model is fine-tuned (more sensitive)"
        ],
        "solutions_in_order": [
            "Try [32, 16, 8, 4] instead of [16, 8, 4]",
            "Use 4 groups instead of 3",
            "Increase calibration size (10k → 25k)",
            "Use PWCCA + CKA together (not single metric)",
            "If still large, use QAT (Quantization-Aware Training) for 1-3 epochs"
        ]
    },
    
    "out_of_memory": {
        "problem": "Out of memory error during execution",
        "causes": [
            "Batch size too large",
            "GPU memory insufficient",
            "Dataset too large for memory"
        ],
        "solutions": [
            "Script auto-selects batch size - check printed value",
            "Reduce batch size manually (edit script)",
            "Use smaller calibration/eval set (5k instead of 25k)",
            "Run on CPU (slower but uses system RAM)"
        ]
    },
    
    "slow_extraction": {
        "problem": "Layer output extraction is very slow",
        "causes": [
            "Large dataset or large batch size",
            "Slow GPU or CPU execution",
            "I/O bottleneck"
        ],
        "solutions": [
            "Use smaller calibration set (5k instead of 25k)",
            "Check GPU utilization (should be > 80%)",
            "Reduce num_workers in DataLoader (edit script)",
            "Close other GPU-consuming processes"
        ]
    },
    
    "files_not_found": {
        "problem": "Sensitivity file not found in Phase 2",
        "causes": [
            "Phase 1 didn't complete successfully",
            "Sensitivities/ directory not created",
            "Wrong file naming"
        ],
        "solutions": [
            "Run Phase 1 again and watch for errors",
            "Manually create Sensitivities/ directory",
            "Check filenames match pattern: layer_sensitivity_BERT_IMDB_*.json",
            "Run Phase 1 and check output folder before Phase 2"
        ]
    },
    
    "inconsistent_results": {
        "problem": "Results vary between runs",
        "causes": [
            "Small calibration set (inherent noise)",
            "Different random samples",
            "Floating point precision"
        ],
        "solutions": [
            "Use larger calibration set (25k recommended)",
            "Use stratified sampling (more stable)",
            "Run multiple times and average results",
            "Use same seed (script sets seed=42 by default)"
        ]
    }
}

# ============================================================================
# QUICK REFERENCE: EXPECTED RESULTS
# ============================================================================

EXPECTED_RESULTS = {
    "non_finetuned_bert": {
        "model": "BERT-base-uncased (pre-trained only)",
        "typical_configs": [
            {
                "name": "Fast (5k cal, 3 groups, [16/8/4])",
                "accuracy_drop": "2-3%",
                "compression": "2.0x",
                "time": "5 min"
            },
            {
                "name": "Balanced (25k cal, 3 groups, [16/8/4])",
                "accuracy_drop": "1-2%",
                "compression": "2.0x",
                "time": "15 min"
            },
            {
                "name": "Conservative (25k cal, 4 groups, [32/16/8/4])",
                "accuracy_drop": "< 1%",
                "compression": "1.5x",
                "time": "30 min"
            }
        ]
    },
    
    "finetuned_bert": {
        "model": "BERT-base-uncased fine-tuned on IMDB",
        "note": "Fine-tuned models are more sensitive to quantization",
        "typical_configs": [
            {
                "name": "Fast (5k cal, 3 groups, [16/8/4])",
                "accuracy_drop": "2-3%",
                "compression": "2.0x",
                "time": "5 min"
            },
            {
                "name": "Balanced (25k cal, 4 groups, [32/16/8/4])",
                "accuracy_drop": "1-2%",
                "compression": "1.5x",
                "time": "15 min"
            },
            {
                "name": "Conservative (25k cal, 4 groups, [32/16/8/4])",
                "accuracy_drop": "< 1%",
                "compression": "1.5x",
                "time": "30 min",
                "note": "May need light QAT if drop > 3%"
            }
        ]
    }
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

USAGE_EXAMPLES = """
Example 1: Quick Test
====================
$ python BERT_base_IMDB.py
> Select: "Run both phases"
> Calibration: "5k (fast, noisy)"
> Sampling: "Stratified sampling (recommended)"
> Metrics: "PWCCA + CKA (recommended)"
> Eval size: "5k (fast iteration)"
> Clustering: "K-means (recommended)"
> Groups: "3 groups (simpler)"
> Bits: "[16, 8, 4]"
Time: ~5 minutes
Result: ~2-3% accuracy drop, 2.0x compression


Example 2: Final Results
========================
$ python BERT_base_IMDB.py
> Select: "Run both phases"
> Calibration: "25k (full IMDB train, recommended)"
> Sampling: "Stratified sampling (recommended)"
> Metrics: "PWCCA + CKA (recommended)"
> Eval size: "25k (full test, recommended)"
> Clustering: "K-means (recommended)"
> Groups: "4 groups (finer control)"
> Bits: "[32, 16, 8, 4]"
Time: ~30 minutes
Result: <1% accuracy drop, 1.5x compression


Example 3: Exploring Different Metrics
=======================================
Run 1:
$ python BERT_base_IMDB.py
> Phase 1 only
> Calibration: 25k
> Metrics: "PWCCA only"

Run 2:
$ python BERT_base_IMDB.py
> Phase 1 only
> Calibration: 25k
> Metrics: "SVCCA only"

Run 3:
$ python BERT_base_IMDB.py
> Phase 2
> Select different sensitivity files for each run
> Compare results

Result: Understand impact of different metrics


Example 4: Aggressive Compression
==================================
$ python BERT_base_IMDB.py
> Select: "Run both phases"
> Calibration: "25k"
> Metrics: "PWCCA + CKA"
> Eval size: "25k"
> Clustering: "Hierarchical clustering"
> Groups: "4 groups"
> Bits: "[8, 4, 2, 2]"
Time: ~30 minutes
Result: ~5-10% accuracy drop, 4.0x compression
WARNING: May need QAT to recover accuracy
"""

# ============================================================================
# PRINT HELPER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BERT_base_IMDB.py - Configuration Reference")
    print("=" * 70)
    print()
    print("Available presets:")
    for name, config in PRESETS.items():
        print(f"  - {name}: {config['description']}")
    print()
    print("See docstrings in this file for detailed documentation.")
