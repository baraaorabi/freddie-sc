import sys
import re
from collections import Counter

if len(config) == 0:

    configfile: "Simulate.config.yaml"


outpath = config["outpath"]
preproc_d = f"{outpath}/preprocess"
TS_d = f"{outpath}/TS"
exprmnts_re = "|".join([re.escape(x) for x in config["TS_experiments"]])


def exprmnt_final_file(exprmnt):
    prefix = [list(step)[0] for step in config["TS_experiments"][exprmnt]["pipeline"]]
    if prefix[-1] in ["Seq"]:
        prefix.append("fastq")
    elif prefix[-1] in [
        "Tsb",
        "Flt",
        "PCR",
        "plA",
        "SCB",
        "Tag",
        "Flp",
        "Trc",
        "Shf",
        "Uns",
    ]:
        prefix.append("mdf")
    else:
        raise ValueError(f"Invalid terminal pipeline step! {prefix[-1]}")
    prefix = ".".join(prefix)
    final_file = f"{TS_d}/{exprmnt}/{prefix}"
    return final_file


def get_sample_ref_names(sample):
    # Check if sample is real
    if sample in config["samples"]:
        return [config["samples"][sample]["ref"]]
    # If not, then sample must be a TS experiment
    if sample in config["TS_experiments"]:
        step = config["TS_experiments"][sample]["pipeline"][0]
        rule_name = list(step)[0]
        step = step[rule_name]
        # If 1st step is transcribe, then return its model's reference
        if rule_name == "Tsb":
            return get_sample_ref_names(step["model"])
        # If 1st step is merge, then return the references of its sources
        if rule_name == "Mrg":
            ref_names = set()
            for source in step["sources"]:
                ref_names.update(get_sample_ref_names(source))
            ref_names = sorted(ref_names)
            return ref_names
        raise ValueError(f"Invalid 1st rule ({rule_name}) for sample ({sample})!")
    raise ValueError(f"Invalid sample ({sample})!")


def get_sample_ref(sample, ref_type):
    ref_names = get_sample_ref_names(sample)
    ref_name = ":".join(ref_names)
    if ref_type in ["DNA", "cDNA"]:
        file_type = "fasta"
    elif ref_type == "GTF":
        file_type = "gtf"
    else:
        raise ValueError(f"Invalid reference type! {ref_type}")
    return f"{preproc_d}/refs/{ref_name}.{ref_type}.{file_type}"


def get_sample_fastqs(name):
    if name in config["samples"]:
        sample = name
        return config["samples"][sample]["fastq"]
    if name in config["TS_experiments"]:
        exprmnt = name
        fastq = exprmnt_final_file(exprmnt)
        assert fastq.endswith(".fastq")
        return [fastq]
    raise ValueError(f"Invalid experiment/sample name! {name}")


def get_step(exprmnt, prefix):
    idx = len(prefix.split(".")) - 1
    step = config["TS_experiments"][exprmnt]["pipeline"][idx]
    rule_name = list(step)[0]
    return step[rule_name]


def get_merge_mdf_input(wc):
    step = get_step(wc.exprmnt, "Mrg")
    mdfs = list()
    for source in step["sources"]:
        mdf = exprmnt_final_file(source)
        assert mdf.endswith(".mdf")
        mdfs.append(mdf)
    return mdfs


def get_sequencer_model_input(wc, model_type):
    step = get_step(wc.exprmnt, f"{wc.prefix}.Seq")
    model = step["model"]
    if model in config["samples"] or model in config["TS_experiments"]:
        return f"{preproc_d}/models/badread/{model}.{model_type}.gz"
    else:
        return list()


def get_kde_model_input(wc):
    step = get_step(wc.exprmnt, f"{wc.prefix}.Kde")
    model = step["model"]
    kde_input = [
        f"{preproc_d}/models/truncate/{model}.{x}.npy"
        for x in ["grid", "X_idxs", "Y_idxs"]
    ]
    return kde_input


rule all:
    input:
        [
            f"{TS_d}/{exprmnt}.fastq"
            for exprmnt in config["TS_experiments"]
            if exprmnt_final_file(exprmnt).endswith("fastq")
        ],


def get_source_mdfs(exprmnt):
    first_step = config["TS_experiments"][exprmnt]["pipeline"][0]
    rule_name = list(first_step)[0]
    first_step = first_step[rule_name]
    if rule_name == "Mrg":
        return sorted(
            {mdf for source in first_step["sources"] for mdf in get_source_mdfs(source)}
        )
    if rule_name == "Tsb":
        return [f"{TS_d}/{exprmnt}/Tsb.mdf"]


rule ground_truth_files:
    input:
        fastq=lambda wc: exprmnt_final_file(wc.exprmnt),
        mdfs=lambda wc: get_source_mdfs(wc.exprmnt),
    output:
        fastq=f"{TS_d}/{{exprmnt}}.fastq",
        tsv=f"{TS_d}/{{exprmnt}}.tsv",
    wildcard_constraints:
        exprmnt=exprmnts_re,
    run:
        shell("cp {input.fastq} {output.fastq}")
        rid_to_mid = dict()
        for idx, line in enumerate(open(input.fastq)):
            if idx % 4 != 0:
                continue
            rid = line.strip().split()[0][1:]
            comments = line.strip().split()[1:]
            for comment in comments:
                if comment.startswith("molecule_id="):
                    mid = comment.split("=")[1]
                    rid_to_mid[rid] = mid
        mid_to_tid = dict()
        mid_to_cb = dict()
        for mdf in input.mdfs:
            for line in open(mdf):
                if line[0] != "+":
                    continue
                line = line.strip().split("\t")
                mid = line[0][1:]
                cb = ""
                tid = ""
                for comment in line[2].split(";"):
                    if comment.startswith("CB"):
                        comment = comment.split("=")
                        if len(comment) == 2:
                            cb = comment[1]
                    elif comment.startswith("tid"):
                        tid = comment.split("=")[1]
                mid_to_tid[mid] = tid
                mid_to_cb[mid] = cb

        outfile = open(output.tsv, "w+")
        print(
            "\t".join(["read_id", "molecule_id", "transcript_id", "cell_barcode"]),
            file=outfile,
        )
        for rid, mid in rid_to_mid.items():
            parent_md = re.split("_|\.", mid)[0]
            record = [
                rid,
                mid,
                mid_to_tid[parent_md],
                mid_to_cb[parent_md],
            ]
            print("\t".join(record), file=outfile)
        outfile.close()


if config["enable_piping"] == True:
    sys.stderr.write("Piping enabled for TKSM!\n")
    merge_source_mdf_counter = Counter()
    merge_to_numbered_sources = dict()

    for exprmnt in config["TS_experiments"]:
        step, details = tuple(config["TS_experiments"][exprmnt]["pipeline"][0].items())[
            0
        ]
        if not step == "Mrg":
            continue
        merge_to_numbered_sources[exprmnt] = list()
        for source_exprmnt in details["sources"]:
            source_mdf = exprmnt_final_file(source_exprmnt)
            merge_to_numbered_sources[exprmnt].append(
                f"{source_mdf}.{merge_source_mdf_counter[source_mdf]}"
            )
            merge_source_mdf_counter[source_mdf] += 1

    for mdf, count in merge_source_mdf_counter.items():

        rule:
            input:
                mdf=mdf,
                script="py/mdf_tee.py",
            output:
                [pipe(f"{mdf}.{number}") for number in range(count)],
            shell:
                "python {input.script} {input.mdf} {output}"

else:

    def pipe(X):
        return X

rule make_tksm:
    output:
        "extern/tksm/build/bin/tksm"
    conda:
        "Snakemake-envs/tksm.yaml"
    threads:
        32
    shell:
        "cd extern/tksm && make -j {threads} && ./install.sh"

rule sequence:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
        fastas=lambda wc: get_sample_ref(wc.exprmnt, "DNA"),
        qscore_model=lambda wc: get_sequencer_model_input(wc, "qscore"),
        error_model=lambda wc: get_sequencer_model_input(wc, "error"),
    output:
        fastq=f"{TS_d}/{{exprmnt}}/{{prefix}}.Seq.fastq",
    threads: 32
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Seq")["params"],
        fastas=lambda wc: get_sample_ref(wc.exprmnt, "DNA"),
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} sequence"
        " -i {input.mdf}"
        " --references {params.fastas}"
        " -o {output.fastq}"
        " --threads {threads}"
        " --badread-error-model={input.error_model}"
        " --badread-qscore-model={input.qscore_model}"
        " {params.other}"


rule filter:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Flt.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Flt")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} filter"
        " -i {input.mdf}"
        " -t {output.mdf}"
        " {params.other}"


rule truncate:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
        kde=get_kde_model_input,
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Trc.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Trc")["params"],
        kde=lambda wc: ",".join(get_kde_model_input(wc)),
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} truncate"
        " -i {input.mdf}"
        " --kde-model={params.kde}"
        " -o {output.mdf}"
        " {params.other}"


rule unsegment:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Uns.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Uns")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} unsegment"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule shuffle:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Shf.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Shf")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} shuffle"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule flip:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Flp.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Flp")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} flip"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule pcr:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.PCR.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.PCR")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} pcr"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule tag:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.Tag.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.Tag")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} tag"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule single_cell_barcoder:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.SCB.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.SCB")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} scb"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


rule polyA:
    input:
        mdf=f"{TS_d}/{{exprmnt}}/{{prefix}}.mdf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/{{prefix}}.plA.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"{wc.prefix}.plA")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} polyA"
        " -i {input.mdf}"
        " -o {output.mdf}"
        " {params.other}"


### Entry rules ###
rule transcribe:
    input:
        tsv=lambda wc: f"{preproc_d}/tksm_abundance/{get_step(wc.exprmnt, 'Tsb')['model']}.{get_step(wc.exprmnt, 'Tsb')['mode']}.tsv",
        gtf=lambda wc: get_sample_ref(wc.exprmnt, "GTF"),
        tksm="extern/tksm/build/bin/tksm",
    output:
        mdf=pipe(f"{TS_d}/{{exprmnt}}/Tsb.mdf"),
    params:
        other=lambda wc: get_step(wc.exprmnt, f"Tsb")["params"],
    wildcard_constraints:
        exprmnt=exprmnts_re,
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} transcribe"
        " -a {input.tsv}"
        " -g {input.gtf}"
        " -o {output.mdf}"
        " {params.other}"


if config["enable_piping"] == False:

    rule merge:
        input:
            mdfs=get_merge_mdf_input,
        output:
            mdf=pipe(f"{TS_d}/{{exprmnt}}/Mrg.mdf"),
        shell:
            "cat {input.mdfs} > {output.mdf}"

else:

    rule merge:
        input:
            script="py/mdf_cat.py",
            mdfs=lambda wc: merge_to_numbered_sources[wc.exprmnt],
        output:
            mdf=pipe(f"{TS_d}/{{exprmnt}}/Mrg.mdf"),
        shell:
            "python {input.script} {input.mdfs}  {output.mdf}"


### Preprocessing rules ###
rule abundance:
    input:
        paf=f"{preproc_d}/minimap2/{{sample}}.cDNA.paf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        tsv=f"{preproc_d}/tksm_abundance/{{sample}}.Xpr.tsv",
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} abundance"
        " -p {input.paf}"
        " -o {output.tsv}"


rule abundance_sc:
    input:
        paf=f"{preproc_d}/minimap2/{{sample}}.cDNA.paf",
        lr_matches=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.lr_matches.tsv.gz",
        tksm="extern/tksm/build/bin/tksm",
    output:
        tsv=f"{preproc_d}/tksm_abundance/{{sample}}.Xpr_sc.tsv",
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} abundance"
        " -p {input.paf}"
        " -m {input.lr_matches}"
        " -o {output.tsv}"


rule model_truncation:
    input:
        paf=f"{preproc_d}/minimap2/{{sample}}.cDNA.paf",
        tksm="extern/tksm/build/bin/tksm",
    output:
        x=f"{preproc_d}/models/truncate/{{sample}}.X_idxs.npy",
        y=f"{preproc_d}/models/truncate/{{sample}}.Y_idxs.npy",
        g=f"{preproc_d}/models/truncate/{{sample}}.grid.npy",
    params:
        out_prefix=f"{preproc_d}/models/truncate/{{sample}}",
    threads: 32
    conda:
        "Snakemake-envs/tksm.yaml"
    shell:
        "{input.tksm} model-truncation"
        " -i {input.paf}"
        " -o {params.out_prefix}"
        " --threads {threads}"


rule minimap_cdna:
    input:
        reads=lambda wc: get_sample_fastqs(wc.sample),
        ref=lambda wc: get_sample_ref(wc.sample, "cDNA"),
    output:
        paf=f"{preproc_d}/minimap2/{{sample}}.cDNA.paf",
    threads: 32
    conda:
        "Snakemake-envs/minimap2.yaml"
    shell:
        "minimap2"
        " -t {threads}"
        " -x map-ont"
        " -p 0.0"
        " -c --eqx"
        " -o {output.paf}"
        " {input.ref}"
        " {input.reads}"


rule scTagger_match:
    input:
        lr_tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
        wl_tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.bc_whitelist.tsv.gz",
    output:
        lr_tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.lr_matches.tsv.gz",
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
        tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
        wl=config["refs"]["10x_bc"],
    output:
        tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.bc_whitelist.tsv.gz",
    conda:
        "Snakemake-envs/sctagger.yaml"
    shell:
        "scTagger.py extract_sr_bc_from_lr"
        " -i {input.tsv}"
        " -wl {input.wl}"
        " -o {output.tsv}"


rule scTagger_lr_seg:
    input:
        reads=lambda wc: get_sample_fastqs(wc.sample),
    output:
        tsv=f"{preproc_d}/scTagger/{{sample}}/{{sample}}.lr_bc.tsv.gz",
    threads: 32
    conda:
        "Snakemake-envs/sctagger.yaml"
    shell:
        "scTagger.py extract_lr_bc"
        " -r {input.reads}"
        " -o {output.tsv}"
        " -t {threads}"


rule minimap_cdna_for_badread_models:
    input:
        reads=lambda wc: get_sample_fastqs(wc.sample),
        ref=lambda wc: get_sample_ref(wc.sample, "cDNA"),
    output:
        paf=f"{preproc_d}/badread/{{sample}}.badread.cDNA.paf",
    threads: 32
    conda:
        "Snakemake-envs/minimap2.yaml"
    shell:
        "minimap2"
        " -t {threads}"
        " -x map-ont"
        " -c"
        " -o {output.paf}"
        " {input.ref}"
        " {input.reads}"


rule badread_error_model:
    input:
        reads=lambda wc: get_sample_fastqs(wc.sample),
        ref=lambda wc: get_sample_ref(wc.sample, "cDNA"),
        paf=f"{preproc_d}/badread/{{sample}}.badread.cDNA.paf",
    output:
        model=f"{preproc_d}/models/badread/{{sample}}.error.gz",
    conda:
        "Snakemake-envs/badread.yaml"
    shell:
        "badread error_model"
        " --reads {input.reads}"
        " --reference {input.ref}"
        " --alignment {input.paf}"
        " --max_alignments 250000"
        " > {output.model}"


rule badread_qscore_model:
    input:
        reads=lambda wc: get_sample_fastqs(wc.sample),
        ref=lambda wc: get_sample_ref(wc.sample, "cDNA"),
        paf=f"{preproc_d}/badread/{{sample}}.badread.cDNA.paf",
    output:
        model=f"{preproc_d}/models/badread/{{sample}}.qscore.gz",
    conda:
        "Snakemake-envs/badread.yaml"
    shell:
        "badread qscore_model"
        " --reads {input.reads}"
        " --reference {input.ref}"
        " --alignment {input.paf}"
        " --max_alignments 250000"
        " > {output.model}"


rule cat_refs:
    input:
        refs=lambda wc: [
            config["refs"][ref_name][wc.ref_type]
            for ref_name in wc.ref_names.split(":")
        ],
    output:
        ref=f"{preproc_d}/refs/{{ref_names}}.{{ref_type}}.{{file_type}}",
    run:
        for ref in input.refs:
            if ref.endswith(".gz"):
                shell("zcat {ref} >> {output.ref}")
            else:
                shell("cat {ref} >> {output.ref}")
