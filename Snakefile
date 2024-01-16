configfile: "config.yaml"


outpath = config["outpath"].rstrip("/")

output_d = f"{outpath}/results"
logs_d = f"{outpath}/logs"


rule all:
    input:
        expand(
            f"{output_d}/{{sample}}/{{sample}}.sorted.bam",
            sample=config["samples"],
        ),
        expand(
            f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_matches.tsv.gz",
            sample=config["samples"],
        ),
        expand(f"{output_d}/{{sample}}/freddie.split", sample=config["samples"]),
        expand(f"{output_d}/{{sample}}/freddie.segment", sample=config["samples"]),
        expand(f"{output_d}/{{sample}}/freddie.cluster", sample=config["samples"]),
        expand(
            f"{output_d}/{{sample}}/freddie.isoforms.gtf",
            sample=config["samples"],
        ),


rule minimap2:
    input:
        reads=lambda wc: config["samples"][wc.sample]["reads"],
        genome=lambda wc: config["refs"][config["samples"][wc.sample]["ref"]]["DNA"],
    output:
        bam=protected(f"{output_d}/{{sample}}/{{sample}}.sorted.bam"),
        bai=protected(f"{output_d}/{{sample}}/{{sample}}.sorted.bam.bai"),
    threads: 32
    conda:
        "Snakemake-envs/minimap2.yaml"
    resources:
        mem="128G",
        time=1439,
    shell:
        "minimap2 -a -x splice -t {threads} {input.genome} {input.reads} | "
        "  samtools sort -T {output.bam}.tmp -m 2G -@ {threads} -O bam - > {output.bam} && "
        "  samtools index {output.bam} "


rule split:
    input:
        script=config["exec"]["split"],
        reads=lambda wc: config["samples"][wc.sample]["reads"],
        bam=f"{output_d}/{{sample}}/{{sample}}.sorted.bam",
    output:
        split=directory(f"{output_d}/{{sample}}/freddie.split"),
    conda:
        "Snakemake-envs/scFreddie.yaml"
    threads: 32
    resources:
        mem="16G",
        time=359,
    shell:
        "{input.script} -b {input.bam} -r {input.reads} -o {output.split} -t {threads}"


rule segment:
    input:
        script=config["exec"]["segment"],
        split=f"{output_d}/{{sample}}/freddie.split",
    output:
        segment=directory(f"{output_d}/{{sample}}/freddie.segment"),
    conda:
        "Snakemake-envs/scFreddie.yaml"
    threads: 32
    resources:
        mem="32G",
        time=599,
    shell:
        "{input.script} -s {input.split} -o {output.segment} -t {threads}"


rule cluster:
    input:
        script=config["exec"]["cluster"],
        segment=f"{output_d}/{{sample}}/freddie.segment",
    output:
        cluster=directory(f"{output_d}/{{sample}}/freddie.cluster"),
        logs=directory(f"{output_d}/{{sample}}/freddie.cluster_logs"),
        log=f"{logs_d}/{{sample}}/freddie.cluster.log",
    params:
        g_license=config["gurobi"]["license"],
        g_timeout=config["gurobi"]["timeout"],
    conda:
        "Snakemake-envs/scFreddie.yaml"
    threads: 32
    resources:
        mem="32G",
        time=999,
    shell:
        "export GRB_LICENSE_FILE={params.g_license}; "
        "{input.script} -s {input.segment} -o {output.cluster} -l {output.logs} -t {threads} -to {params.g_timeout} > {output.log}"


rule isoforms:
    input:
        script=config["exec"]["isoforms"],
        split=f"{output_d}/{{sample}}/freddie.split",
        cluster=f"{output_d}/{{sample}}/freddie.cluster",
    output:
        isoforms=protected(f"{output_d}/{{sample}}/freddie.isoforms.gtf"),
    threads: 8
    resources:
        mem="16G",
        time=359,
    shell:
        "{input.script} -s {input.split} -c {input.cluster} -o {output.isoforms} -t {threads}"


rule scTagger_match:
    input:
        lr_tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
        wl_tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.bc_whitelist.tsv.gz",
    output:
        lr_tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_matches.tsv.gz",
    threads: 32
    conda:
        "Snakemake-envs/sctagger.yaml"
    shell:
        "scTagger.py match_trie"
        " -lr {input.lr_tsv}"
        " -sr {input.wl_tsv}"
        " -o {output.lr_tsv}"
        " -t {threads}"


rule scTagger_extract_bc:
    input:
        tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
        wl= lambda wc: config["refs"]["barcodes"][config["samples"][wc.sample]["bc_wl"]],
    output:
        tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.bc_whitelist.tsv.gz",
    conda:
        "Snakemake-envs/sctagger.yaml"
    shell:
        "scTagger.py extract_sr_bc_from_lr"
        " -i {input.tsv}"
        " -wl {input.wl}"
        " -o {output.tsv}"


rule scTagger_lr_seg:
    input:
        reads=lambda wc: config["samples"][wc.sample]["reads"],
    output:
        tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
    threads: 32
    conda:
        "Snakemake-envs/sctagger.yaml"
    shell:
        "scTagger.py extract_lr_bc"
        " -r {input.reads}"
        " -o {output.tsv}"
        " -t {threads}"
