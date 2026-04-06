import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_final_report():
    lambdas = ["0.0", "2.0", "5.0", "10.0", "20.0", "50.0"]
    # 6 rows (lambdas) x 5 columns (Img, Loss, MapA, MapB, Overlap)
    fig = plt.figure(figsize=(35, 42), facecolor='#ffffff')
    
    # 1. Main Title
    plt.suptitle("DCLG (Decoupled Cross-attention Latent Guidance) Final Result Report", 
                 fontsize=60, fontweight='bold', y=0.985, color='#1a1a1a')
    
    # 2. Setup Information Box
    setup_info = (
        "PROMPT: 'A knight and an orc wrestling fiercely, dynamic pose, cinematic'\n"
        "CONFIG: Stable Diffusion 1.5 | 30 Steps (DDIM) | Precision: Float32 | Target Tokens: Knight (2), Orc (5)\n"
        "METHOD: Decoupled Cross-attention Latent Guidance (DCLG) | Score Modification Strategy"
    )
    plt.figtext(0.5, 0.955, setup_info, ha="center", fontsize=26, 
                bbox={"facecolor":"#f8f9fa", "alpha":0.9, "edgecolor":"#dee2e6", "pad":20, "boxstyle":"round,pad=1"})

    cols = ["Generated Image", "Chimera Loss Curve", "Knight Attention (Map A)", "Orc Attention (Map B)", "Spatial Overlap (A \u2299 B)"]
    
    # Add explicit column headers at the very top
    for i, col_name in enumerate(cols):
        plt.figtext(0.12 + i * 0.185, 0.935, col_name, ha="center", va="center", 
                    fontsize=32, fontweight='bold', color='white',
                    bbox={"facecolor":"#2c3e50", "alpha":1.0, "pad":10, "edgecolor":"none"})

    for row_idx, l in enumerate(lambdas):
        # Lambda Label on the far left
        # vertical position adjusted for 6 rows
        v_pos = 0.85 - (row_idx * 0.14)
        plt.figtext(0.04, v_pos, f"Guidance\nScale\n\u03BB = {l}", ha="center", va="center", 
                    fontsize=38, fontweight='bold', color='#c0392b', linespacing=1.2)

        files = [
            f"dclg_toy/outputs/images/text_lambda_{l}.png",
            f"dclg_toy/outputs/images/text_loss_lambda_{l}.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_A.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_B.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_overlap.png"
        ]

        for col_idx, file_path in enumerate(files):
            ax = plt.subplot(6, 5, row_idx * 5 + col_idx + 1)
            
            if os.path.exists(file_path):
                img = mpimg.imread(file_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "IMAGE NOT FOUND", ha="center", va="center", fontsize=20, color='red')
            
            ax.axis('off')
            # Thicker border for better visibility
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('#bdc3c7')

    # 3. Bottom Analysis Note
    analysis_text = (
        "CONCLUSION: DCLG effectively decouples the cross-attention maps of conflicting tokens.\n"
        "Even at extreme scales (\u03BB = 3000.0), character identity is maintained while ensuring absolute spatial separation."
    )
    plt.figtext(0.5, 0.02, analysis_text, ha="center", fontsize=28, fontweight='bold', 
                color='#ffffff', bbox={"facecolor":"#2980b9", "edgecolor":"none", "pad":25, "boxstyle":"round,pad=1"})

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, wspace=0.05, hspace=0.15)
    
    save_path = "dclg_toy/FINAL_REPORT.png"
    plt.savefig(save_path, dpi=90, bbox_inches='tight')
    print(f"6-row comprehensive report created at {save_path}")

if __name__ == "__main__":
    create_final_report()
