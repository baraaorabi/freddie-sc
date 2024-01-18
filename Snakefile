configfile: "config.yaml"


from collections import defaultdict
import gzip

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
            f"{output_d}/rname_to_celltypes/{{sample}}.tsv",
            sample=config["samples"],
        ),
        expand(
            f"{output_d}/{{sample}}/freddie.isoforms.gtf",
            sample=config["samples"],
        ),


rule minimap2:
    input:
        reads=lambda wc: config["samples"][wc.sample]["reads"],
        genome=lambda wc: config["refs"][config["samples"][wc.sample]["ref"]]["DNA"],
    output:
        bam=f"{output_d}/{{sample}}/{{sample}}.sorted.bam",
        bai=f"{output_d}/{{sample}}/{{sample}}.sorted.bam.bai",
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


rule freddie:
    input:
        script="freddie.py",
        # reads=lambda wc: config["samples"][wc.sample]["reads"],
        bam=f"{output_d}/{{sample}}/{{sample}}.sorted.bam",
        rname_to_celltypes=f"{output_d}/rname_to_celltypes/{{sample}}.tsv",
    output:
        isoforms=f"{output_d}/{{sample}}/freddie.isoforms.gtf",
    threads: 8
    resources:
        mem="16G",
        time=359,
    shell:
        "./{input.script}"

        " --rname-to-celltypes {input.rname_to_celltypes}"
        " --bam {input.bam}"
        " --output {output.isoforms}"
        " --threads {threads}"


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
        wl=lambda wc: config["refs"]["barcodes"][config["samples"][wc.sample]["bc_wl"]],
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


rule rname_to_celltypes:
    input:
        lr_matches_tsv=f"{output_d}/scTagger/{{sample}}/{{sample}}.lr_matches.tsv.gz",
        truth_tsv=lambda wc: config["samples"][wc.sample]["truth"],
    output:
        tsv=f"{output_d}/rname_to_celltypes/{{sample}}.tsv",
    run:
        cb_to_celltypes = defaultdict(set)
        with open(input.truth_tsv, "r") as f:
            f.readline()
            for line in f:
                rname, mid, tid, cb, ct = line.strip().split()
                if cb == ".":
                    continue
                cb_to_celltypes[cb].update(ct.split(","))

        rname_to_celltypes = defaultdict(set)
        with gzip.open(input.lr_matches_tsv, "rt") as f:
            for line in f:
                rname, dist, cnt, seg, barcodes = line.strip().split()
                for bc in barcodes.split(","):
                    rname_to_celltypes[rname].update(cb_to_celltypes[bc])

        with open(output.tsv, "w+") as f:
            for rname, celltypes in rname_to_celltypes.items():
                f.write(f"{rname}\t{','.join(celltypes)}\n")
