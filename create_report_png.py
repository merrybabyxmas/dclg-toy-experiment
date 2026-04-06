import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_report_png():
    lambdas = ["0.0", "50.0", "200.0", "1000.0"]
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    
    # Title
    plt.suptitle("DCLG Toy Experiment: Text-to-Image Chimera Suppression", fontsize=40, fontweight='bold', y=0.96)
    
    # Subtitle / Setup Info
    setup_text = (
        "Model: SD 1.5 (float32) | Steps: 30 (DDIM) | Prompt: 'A knight and an orc wrestling fiercely...'\n"
        "Guidance: Decoupled Cross-attention Latent Guidance (DCLG) | Loss: Mean Overlap + Erasure Penalty"
    )
    plt.figtext(0.5, 0.91, setup_text, ha="center", fontsize=20, bbox={"facecolor":"orange", "alpha":0.2, "pad":10})

    # Rows: 1. Generated Images, 2. Overlap Maps
    for i, l in enumerate(lambdas):
        # 1. Generated Images
        ax1 = plt.subplot(2, 4, i + 1)
        img_path = f"dclg_toy/outputs/images/text_lambda_{l}.png"
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax1.imshow(img)
            ax1.set_title(f"Lambda = {l}\n(Generated Image)", fontsize=22, fontweight='bold')
        ax1.axis('off')

        # 2. Overlap Maps (Step 15)
        ax2 = plt.subplot(2, 4, i + 5)
        map_path = f"dclg_toy/outputs/attention_maps/step15_lambda{l}_overlap.png"
        if os.path.exists(map_path):
            map_img = mpimg.imread(map_path)
            ax2.imshow(map_img)
            ax2.set_title(f"Lambda = {l}\n(Overlap Map)", fontsize=22, fontweight='bold')
        ax2.axis('off')

    # Add Legend/Conclusion at the bottom
    conclusion_text = (
        "Analysis: As Lambda increases, the Overlap Map activation area significantly decreases.\n"
        "The Knight and Orc attention regions become spatially separated, effectively preventing 'Chimera' artifacts."
    )
    plt.figtext(0.5, 0.04, conclusion_text, ha="center", fontsize=24, fontweight='bold', 
                color='darkblue', bbox={"facecolor":"lightcyan", "edgecolor":"blue", "pad":15})

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    plt.savefig("dclg_toy/REPORT.png", dpi=150, bbox_inches='tight')
    print("Report PNG created successfully at dclg_toy/REPORT.png")

if __name__ == "__main__":
    create_report_png()
