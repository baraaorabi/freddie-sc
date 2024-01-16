import sys
import re
from collections import Counter

if len(config) == 0:

    configfile: "Simulate.config.yaml"


module TS_smk:
    snakefile:
        "extern/tksm/Snakefile"
    config:
        config


### Import TKSM Snakemake rules
use rule * from TS_smk exclude all


outpath = config["outpath"]
preproc_d = f"{outpath}/preprocess"
TS_d = f"{outpath}/TS"
exprmnts_re = "|".join([re.escape(x) for x in config["TS_experiments"]])
fastq_exprmnts = [
    x
    for x in config["TS_experiments"]
    if TS_smk.exprmnt_final_file(x).endswith(".fastq")
]
fastq_exprmnts_re = "|".join([re.escape(x) for x in fastq_exprmnts])


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


rule all:
    input:
        expand(
            f"{TS_d}/{{exprmnt}}.{{ext}}",
            exprmnt=fastq_exprmnts,
            ext=["fastq", "tsv"],
        ),
    default_target: True


rule ground_truth_files:
    input:
        fastq=lambda wc: TS_smk.exprmnt_final_file(wc.exprmnt),
        mdfs=lambda wc: get_source_mdfs(wc.exprmnt),
    output:
        fastq=f"{TS_d}/{{exprmnt}}.fastq",
        tsv=f"{TS_d}/{{exprmnt}}.tsv",
    wildcard_constraints:
        exprmnt=fastq_exprmnts_re,
    run:
        shell(f"cp {input.fastq} {output.fastq}")
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
            assert rid in rid_to_mid
        mid_to_tid = dict()
        mid_to_cb = dict()
        cb_to_cell_lines = dict()
        for mdf in input.mdfs:
            exprmnt = mdf.split("/")[-2]
            for line in open(mdf):
                if line[0] != "+":
                    continue
                line = line.strip().split("\t")
                mid = line[0][1:]
                cb = "."
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
                if cb not in cb_to_cell_lines:
                    cb_to_cell_lines[cb] = set()
                cb_to_cell_lines[cb].add(exprmnt)

        outfile = open(output.tsv, "w+")
        print(
            "\t".join(
                [
                    "read_id",
                    "molecule_id",
                    "transcript_id",
                    "cell_barcode",
                    "cell_lines",
                ]
            ),
            file=outfile,
        )
        for rid, mid in rid_to_mid.items():
            parent_md = re.split("_|\.", mid)[0]
            record = [
                rid,
                mid,
                mid_to_tid[parent_md],
                mid_to_cb[parent_md],
                ",".join(sorted(cb_to_cell_lines[mid_to_cb[parent_md]])),
            ]
            print("\t".join(record), file=outfile)
        outfile.close()
