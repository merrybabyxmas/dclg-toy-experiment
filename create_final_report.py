import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_final_report():
    lambdas = ["0.0", "10.0", "50.0", "100.0", "200.0", "300.0"]
    # 6 rows x 6 columns (Img, OverlayA, OverlayB, MapA, MapB, Overlap)
    fig = plt.figure(figsize=(42, 42), facecolor='#ffffff')
    
    plt.suptitle("DCLG Final Report: Grounding Analysis (Text-Visual Connection)", 
                 fontsize=60, fontweight='bold', y=0.985, color='#1a1a1a')
    
    setup_info = (
        "PROMPT: 'A knight in red armor and an orc with green skin wrestling fiercely...'\n"
        "Grounding Check: Red Overlay = Knight Token, Blue Overlay = Orc Token\n"
        "Visual Proof: Observe how the red/blue highlight accurately tracks the specific character as Lambda increases."
    )
    plt.figtext(0.5, 0.94, setup_info, ha="center", fontsize=26, 
                bbox={"facecolor":"#f8f9fa", "alpha":0.9, "edgecolor":"#dee2e6", "pad":20, "boxstyle":"round,pad=1"})

    cols = ["Generated Image", "Knight Overlay", "Orc Overlay", "Knight Map", "Orc Map", "Spatial Overlap"]
    
    for i, col_name in enumerate(cols):
        plt.figtext(0.13 + i * 0.155, 0.91, col_name, ha="center", va="center", 
                    fontsize=32, fontweight='bold', color='white',
                    bbox={"facecolor":"#2c3e50", "alpha":1.0, "pad":10, "edgecolor":"none"})

    for row_idx, l in enumerate(lambdas):
        v_pos = 0.85 - (row_idx * 0.14)
        plt.figtext(0.04, v_pos, f"Scale\n\u03BB = {l}", ha="center", va="center", 
                    fontsize=38, fontweight='bold', color='#c0392b')

        files = [
            f"dclg_toy/outputs/images/text_lambda_{l}.png",
            f"dclg_toy/outputs/images/text_lambda_{l}_overlay_knight.png",
            f"dclg_toy/outputs/images/text_lambda_{l}_overlay_orc.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_A.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_B.png",
            f"dclg_toy/outputs/attention_maps/step15_lambda{l}_overlap.png"
        ]

        for col_idx, file_path in enumerate(files):
            ax = plt.subplot(6, 6, row_idx * 6 + col_idx + 1)
            if os.path.exists(file_path):
                img = mpimg.imread(file_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
            ax.axis('off')
            rect = plt.Rectangle((0,0), 1, 1, fill=False, color="#bdc3c7", transform=ax.transAxes, linewidth=2)
            ax.add_patch(rect)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.05, wspace=0.05, hspace=0.15)
    plt.savefig("dclg_toy/FINAL_REPORT.png", dpi=90, bbox_inches='tight')

if __name__ == "__main__":
    create_final_report()
