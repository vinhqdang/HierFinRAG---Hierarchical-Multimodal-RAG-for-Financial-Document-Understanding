import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#e0f7fa'):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.01, y-0.01), width, height, boxstyle="round,pad=0.02", 
                                   linewidth=0, facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", 
                                linewidth=1.5, edgecolor='#006064', facecolor=color)
    ax.add_patch(box)
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold', color='#006064')
    return x+width, y+height/2

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#455a64'))

def generate_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Phase 1: Indexing
    ax.text(1.5, 5.5, "Phase 1: Structure-Aware Indexing", fontsize=12, fontweight='bold', color='#1a237e')
    
    draw_box(ax, 0.5, 4.0, 1.5, 0.8, "PDF/JSON\nDocuments")
    draw_arrow(ax, 2.0, 4.4, 2.5, 4.4)
    
    draw_box(ax, 2.5, 4.0, 2.0, 0.8, "Graph Builder\n(Nodes & Edges)", color='#c5cae9')
    draw_arrow(ax, 4.5, 4.4, 5.0, 4.4)
    
    draw_box(ax, 5.0, 4.0, 2.0, 0.8, "Table-Text Graph\n(Heterogeneous)", color='#c5cae9')

    # Phase 2: Retrieval
    ax.text(1.5, 3.0, "Phase 2: Hierarchical Retrieval", fontsize=12, fontweight='bold', color='#1a237e')
    
    draw_box(ax, 0.5, 1.5, 1.5, 0.8, "User Query", color='#fff9c4')
    draw_arrow(ax, 2.0, 1.9, 2.5, 1.9)
    
    draw_box(ax, 2.5, 1.5, 2.0, 0.8, "Hierarchical Search\n(Section -> Para/Table)", color='#ffcc80')
    draw_arrow(ax, 4.5, 1.9, 5.0, 1.9)
    
    draw_box(ax, 5.0, 1.5, 2.0, 0.8, "TTGNN Reranking\n(Structure Aware)", color='#ffcc80')
    draw_arrow(ax, 7.0, 1.9, 7.5, 1.9)

    # Phase 3: Fusion
    ax.text(7.5, 5.5, "Phase 3: Reasoning", fontsize=12, fontweight='bold', color='#1a237e')
    
    # Connecting Arrows across phases
    draw_arrow(ax, 6.0, 4.0, 6.0, 2.3) # Graph to TTGNN
    
    draw_box(ax, 7.5, 1.5, 2.0, 2.0, "Symbolic-Neural\nFusion Router", color='#dcedc8')
    
    # Outputs
    draw_arrow(ax, 9.5, 2.5, 10.0, 3.5) # Generic out direction visualization
    
    plt.tight_layout()
    plt.savefig('Fig0_Architecture.png', dpi=300, bbox_inches='tight')
    print("Diagram generated: Fig0_Architecture.png")

if __name__ == "__main__":
    generate_diagram()
