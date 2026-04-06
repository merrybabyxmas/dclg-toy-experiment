import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_final_report():
    lambdas = ["0.0", "5.0", "10.0", "20.0", "50.0", "100.0", "200.0", "300.0"]
    # 8 rows (lambdas) x 5 columns (Img, Loss, MapA, MapB, Overlap)
    fig = plt.figure(figsize=(35, 56), facecolor='#ffffff')
    
    plt.suptitle("DCLG (Decoupled Cross-attention Latent Guidance) Final Result Report: Text-Only Edition", 
                 fontsize=60, fontweight='bold', y=0.988, color='#1a1a1a')
    
    setup_info = (
        "PROMPT: 'A knight and an orc wrestling fiercely, dynamic pose, cinematic'\n"
        "CONFIG: Stable Diffusion 1.5 | 30 Steps (DDIM) | Float32 | Target Tokens: Knight (2), Orc (5)\n"
        "METHOD: Decoupled Cross-attention Latent Guidance (DCLG) | Score Modification Strategy"
    )
    plt.figtext(0.5, 0.965, setup_info, ha="center", fontsize=26, 
                bbox={"facecolor":"#f8f9fa", "alpha":0.9, "edgecolor":"#dee2e6", "pad":20, "boxstyle":"round,pad=1"})

    cols = ["Generated Image", "Chimera Loss Curve", "Knight Attention (Map A)", "Orc Attention (Map B)", "Spatial Overlap (A \u2299 B)"]
    
    for i, col_name in enumerate(cols):
        plt.figtext(0.12 + i * 0.185, 0.95, col_name, ha="center", va="center", 
                    fontsize=32, fontweight='bold', color='white',
                    bbox={"facecolor":"#2c3e50", "alpha":1.0, "pad":10, "edgecolor":"none"})

    for row_idx, l in enumerate(lambdas):
        # Adjusted v_pos for 8 rows
        v_pos = 0.90 - (row_idx * 0.11)
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
            ax = plt.subplot(8, 5, row_idx * 5 + col_idx + 1)
            if os.path.exists(file_path):
                img = mpimg.imread(file_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('#bdc3c7')

    analysis_text = (
        "CONCLUSION: Text Cross-Attention typically lacks strong spatial grounding for specific entities.\n"
        "Guidance scales up to 300.0 demonstrate the stability of the score modification strategy even under high stress."
    )
    plt.figtext(0.5, 0.015, analysis_text, ha="center", fontsize=28, fontweight='bold', 
                color='#ffffff', bbox={"facecolor":"#2980b9", "edgecolor":"none", "pad":25, "boxstyle":"round,pad=1"})

    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.04, wspace=0.05, hspace=0.15)
    plt.savefig("dclg_toy/FINAL_REPORT.png", dpi=80, bbox_inches='tight')

if __name__ == "__main__":
    create_final_report()
