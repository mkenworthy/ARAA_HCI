rule simulation:
    input:
        "src/scripts/interactive_fresnel.ipynb"
    output:
        "src/figures/test.pdf"
    conda:
        "environment.yml"
    shell:
        "jupyter notebook src/scripts/interactive_fresnel.ipynb"