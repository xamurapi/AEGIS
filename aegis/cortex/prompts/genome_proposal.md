Propose one genome for the next generation.

Gene specification (name: type, range, default):
{genome_spec}

Current champion genome: {champion}
Champion fitness: {champion_fitness}
Recent lineage (genome -> fitness): {lineage}

Every value must lie inside the declared range. A proposal that names an unknown
gene or leaves a range is discarded and replaced by a coordinate mutation.

Respond with ONLY a JSON object:
{
  "genome": {"<gene>": <value>, ...},
  "rationale": "what this variant is testing"
}
