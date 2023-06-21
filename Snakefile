configfile: "config.yaml"


outpath = config["outpath"].rstrip("/")

output_d = "{}/results".format(outpath)
logs_d = "{}/logs".format(outpath)


rule all:
    input:
        expand(
            "{}/{{sample}}/{{sample}}.sorted.bam".format(output_d),
            sample=config["samples"],
        ),
        expand("{}/{{sample}}/freddie.split".format(output_d), sample=config["samples"]),
        expand(
            "{}/{{sample}}/freddie.segment".format(output_d), sample=config["samples"]
        ),
        expand(
            "{}/{{sample}}/freddie.cluster".format(output_d), sample=config["samples"]
        ),
        expand(
            "{}/{{sample}}/freddie.isoforms.gtf".format(output_d),
            sample=config["samples"],
        ),


rule minimap2:
    input:
        reads=lambda wildcards: config["samples"][wildcards.sample]["reads"],
        genome=lambda wildcards: config["references"][
            config["samples"][wildcards.sample]["ref"]
        ]["genome"],
    output:
        bam=protected("{}/{{sample}}/{{sample}}.sorted.bam".format(output_d)),
        bai=protected("{}/{{sample}}/{{sample}}.sorted.bam.bai".format(output_d)),
    threads: 32
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
        reads=lambda wildcards: config["samples"][wildcards.sample]["reads"],
        bam="{}/{{sample}}/{{sample}}.sorted.bam".format(output_d),
    output:
        split=directory("{}/{{sample}}/freddie.split".format(output_d)),
    threads: 32
    resources:
        mem="16G",
        time=359,
    shell:
        "{input.script} -b {input.bam} -r {input.reads} -o {output.split} -t {threads}"


rule segment:
    input:
        script=config["exec"]["segment"],
        split="{}/{{sample}}/freddie.split".format(output_d),
    output:
        segment=directory("{}/{{sample}}/freddie.segment".format(output_d)),
    threads: 32
    resources:
        mem="32G",
        time=599,
    shell:
        "{input.script} -s {input.split} -o {output.segment} -t {threads}"


rule cluster:
    input:
        script=config["exec"]["cluster"],
        segment="{}/{{sample}}/freddie.segment".format(output_d),
    output:
        cluster=directory("{}/{{sample}}/freddie.cluster".format(output_d)),
        logs=directory("{}/{{sample}}/freddie.cluster_logs".format(output_d)),
        log="{}/{{sample}}/freddie.cluster.log".format(logs_d),
    params:
        g_license=config["gurobi"]["license"],
        g_timeout=config["gurobi"]["timeout"],
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
        split="{}/{{sample}}/freddie.split".format(output_d),
        cluster="{}/{{sample}}/freddie.cluster".format(output_d),
    output:
        isoforms=protected("{}/{{sample}}/freddie.isoforms.gtf".format(output_d)),
    threads: 8
    resources:
        mem="16G",
        time=359,
    shell:
        "{input.script} -s {input.split} -c {input.cluster} -o {output.isoforms} -t {threads}"
