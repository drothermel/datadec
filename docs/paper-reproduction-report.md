# Paper verification report

- Paper identity: `arxiv:2504.11393v2`
- Selected run ID: `20260821T141605-qualified`
- Manifest SHA256: `6b2fded0a35bd209fd99effc542c7176999fc9114c17ab4fce2c3f6325c22d86`

## Pinned run identities

| Identity | ID | Digest / state |
| --- | --- | --- |
| Paper | arxiv:2504.11393v2 | SHA256=20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1 |
| Reproduction config | configs/paper\_reproduction.toml | SHA256=55b2a956c94d1d2625a5e7aa02f9d34010c17714eb3d19931b54ab5cf4630f84 |
| Claim registry | docs/paper/claims.toml | SHA256=172e0bf01fb2f4e56750fc9cae89129feef21e7d534d406feaf2601b291654e6 |
| Code | 5bddcd2fac10c0b8ae80dd254eafafd93125c7dc | tree=clean; dirty diff artifact=— |
| Observations | observations.json | SHA256=093d52f2d221ca8cb2538028be7bbc930c75daecfca5d196e57ec53d4006962f; count=442 |

## Evidence and method interpretation

The required evidence boundary is the static claim target; the actual evidence boundary records what this selected run reached. Method provenance records whether a method is paper-derived, upstream-informed, or artifact-derived; provenance does not by itself establish independence. A `source_only_match` confirms only source or author-artifact agreement and is not an independent reproduction. Blocked and contradicted verdicts are successful scientific outcomes, not process failures. This report renders the selected observations as recorded and does not recompute scientific results. Full per-claim details remain in the immutable selected-run observations identity: run `20260821T141605-qualified`, file `observations.json`, SHA256 `093d52f2d221ca8cb2538028be7bbc930c75daecfca5d196e57ec53d4006962f`, 251325 bytes.

## Summary counts

| Dimension | Value | Count |
| --- | --- | ---: |
| Verdict | contradicted | 15 |
| Verdict | source_only_match | 154 |
| Verdict | blocked_missing_input | 64 |
| Verdict | blocked_unspecified_method | 108 |
| Verdict | external_or_citation_dependent | 39 |
| Verdict | not_attempted | 62 |
| Actual evidence boundary | paper_or_final_artifact | 160 |
| Actual evidence boundary | author_downstream_table | 9 |
| Actual evidence boundary | none | 273 |

## Known contradictions and inconsistencies

**These selected-run results contradict a claim or expose an internal inconsistency and must remain visible.**

| ID | Static claim and locator | Expected | Observed / diagnostics | Verdict | Evidence boundary | Counts | Method / policy / verifier | Blocker | Artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD-0269 | All models have sequence length 2024.<br>docs/paper/example\_paper.tex:489-489 | "sequence\_length = 2024" | value="2048"; diagnostics=\["suite fact sequence\_length: expected '2024', observed '2048'","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0276 | 4M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 4M\|32\|64\|1.4e-02\|3.7M\|8\|8\|5,725\|0.4B.<br>docs/paper/tables/suite\_stats.tex:5-5 | "4M\|32\|64\|1.4e-02\|3.7M\|8\|8\|5,725\|0.4B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"64","layers":"8","learning\_rate":"1.4e-02","model\_name":"4M","model\_size":"3.7M","tokens\_trained":"0.4B","training\_steps":"5,715"}; diagnostics=\["training\_steps: expected 5,725, observed 5,715","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0277 | 6M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 6M\|32\|96\|1.2e-02\|6.0M\|8\|8\|9,182\|0.6B.<br>docs/paper/tables/suite\_stats.tex:6-6 | "6M\|32\|96\|1.2e-02\|6.0M\|8\|8\|9,182\|0.6B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"96","layers":"8","learning\_rate":"1.2e-02","model\_name":"6M","model\_size":"6.0M","tokens\_trained":"0.6B","training\_steps":"9,172"}; diagnostics=\["training\_steps: expected 9,182, observed 9,172","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0278 | 8M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 8M\|32\|128\|1.1e-02\|8.5M\|8\|8\|13,039\|0.9B.<br>docs/paper/tables/suite\_stats.tex:7-7 | "8M\|32\|128\|1.1e-02\|8.5M\|8\|8\|13,039\|0.9B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"128","layers":"8","learning\_rate":"1.1e-02","model\_name":"8M","model\_size":"8.5M","tokens\_trained":"0.9B","training\_steps":"13,029"}; diagnostics=\["training\_steps: expected 13,039, observed 13,029","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0279 | 10M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 10M\|32\|144\|1.0e-02\|9.9M\|8\|8\|15,117\|1.0B.<br>docs/paper/tables/suite\_stats.tex:8-8 | "10M\|32\|144\|1.0e-02\|9.9M\|8\|8\|15,117\|1.0B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"144","layers":"8","learning\_rate":"1.0e-02","model\_name":"10M","model\_size":"9.9M","tokens\_trained":"1.0B","training\_steps":"15,107"}; diagnostics=\["training\_steps: expected 15,117, observed 15,107","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0280 | 14M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 14M\|32\|192\|9.2e-03\|14.4M\|8\|8\|21,953\|1.4B.<br>docs/paper/tables/suite\_stats.tex:9-9 | "14M\|32\|192\|9.2e-03\|14.4M\|8\|8\|21,953\|1.4B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"192","layers":"8","learning\_rate":"9.2e-03","model\_name":"14M","model\_size":"14.4M","tokens\_trained":"1.4B","training\_steps":"21,943"}; diagnostics=\["training\_steps: expected 21,953, observed 21,943","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0281 | 16M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 16M\|32\|208\|8.9e-03\|16.0M\|8\|8\|24,432\|1.6B.<br>docs/paper/tables/suite\_stats.tex:10-10 | "16M\|32\|208\|8.9e-03\|16.0M\|8\|8\|24,432\|1.6B" | value={"batch\_size":"32","heads":"8","hidden\_dimension":"208","layers":"8","learning\_rate":"8.9e-03","model\_name":"16M","model\_size":"16.0M","tokens\_trained":"1.6B","training\_steps":"24,422"}; diagnostics=\["training\_steps: expected 24,432, observed 24,422","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0282 | 20M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 20M\|64\|192\|8.4e-03\|19.1M\|8\|16\|14,584\|1.9B.<br>docs/paper/tables/suite\_stats.tex:11-11 | "20M\|64\|192\|8.4e-03\|19.1M\|8\|16\|14,584\|1.9B" | value={"batch\_size":"64","heads":"8","hidden\_dimension":"192","layers":"16","learning\_rate":"8.4e-03","model\_name":"20M","model\_size":"19.1M","tokens\_trained":"1.9B","training\_steps":"14,574"}; diagnostics=\["training\_steps: expected 14,584, observed 14,574","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0283 | 60M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 60M\|96\|384\|5.8e-03\|57.1M\|12\|16\|29,042\|5.7B.<br>docs/paper/tables/suite\_stats.tex:12-12 | "60M\|96\|384\|5.8e-03\|57.1M\|12\|16\|29,042\|5.7B" | value={"batch\_size":"96","heads":"12","hidden\_dimension":"384","layers":"16","learning\_rate":"5.8e-03","model\_name":"60M","model\_size":"57.1M","tokens\_trained":"5.7B","training\_steps":"29,032"}; diagnostics=\["training\_steps: expected 29,042, observed 29,032","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0284 | 90M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 90M\|160\|528\|4.9e-03\|97.9M\|12\|16\|29,901\|9.8B.<br>docs/paper/tables/suite\_stats.tex:13-13 | "90M\|160\|528\|4.9e-03\|97.9M\|12\|16\|29,901\|9.8B" | value={"batch\_size":"160","heads":"12","hidden\_dimension":"528","layers":"16","learning\_rate":"4.9e-03","model\_name":"90M","model\_size":"97.9M","tokens\_trained":"9.8B","training\_steps":"29,891"}; diagnostics=\["training\_steps: expected 29,901, observed 29,891","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0285 | 150M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 150M\|192\|768\|4.2e-03\|151.9M\|12\|12\|38,157\|15.0B.<br>docs/paper/tables/suite\_stats.tex:14-14 | "150M\|192\|768\|4.2e-03\|151.9M\|12\|12\|38,157\|15.0B" | value={"batch\_size":"192","heads":"12","hidden\_dimension":"768","layers":"12","learning\_rate":"4.2e-03","model\_name":"150M","model\_size":"151.9M","tokens\_trained":"15.0B","training\_steps":"38,147"}; diagnostics=\["training\_steps: expected 38,157, observed 38,147","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0286 | 300M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 300M\|320\|1,024\|3.3e-03\|320.0M\|16\|16\|45,787\|30.0B.<br>docs/paper/tables/suite\_stats.tex:15-15 | "300M\|320\|1,024\|3.3e-03\|320.0M\|16\|16\|45,787\|30.0B" | value={"batch\_size":"320","heads":"16","hidden\_dimension":"1,024","layers":"16","learning\_rate":"3.3e-03","model\_name":"300M","model\_size":"320.0M","tokens\_trained":"30.0B","training\_steps":"45,777"}; diagnostics=\["training\_steps: expected 45,787, observed 45,777","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0287 | 530M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 530M\|448\|1,344\|2.8e-03\|530.1M\|16\|16\|57,786\|53.0B.<br>docs/paper/tables/suite\_stats.tex:16-16 | "530M\|448\|1,344\|2.8e-03\|530.1M\|16\|16\|57,786\|53.0B" | value={"batch\_size":"448","heads":"16","hidden\_dimension":"1,344","layers":"16","learning\_rate":"2.8e-03","model\_name":"530M","model\_size":"530.1M","tokens\_trained":"53.0B","training\_steps":"57,766"}; diagnostics=\["training\_steps: expected 57,786, observed 57,766","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0288 | 750M row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 750M\|576\|1,536\|2.5e-03\|681.3M\|16\|16\|63,589\|75.0B.<br>docs/paper/tables/suite\_stats.tex:17-17 | "750M\|576\|1,536\|2.5e-03\|681.3M\|16\|16\|63,589\|75.0B" | value={"batch\_size":"576","heads":"16","hidden\_dimension":"1,536","layers":"16","learning\_rate":"2.5e-03","model\_name":"750M","model\_size":"681.3M","tokens\_trained":"75.0B","training\_steps":"63,579"}; diagnostics=\["training\_steps: expected 63,589, observed 63,579","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |
| DD-0289 | 1B row: model name \| batch size \| hidden dimension \| learning rate \| model size \| heads \| layers \| training steps \| tokens trained = 1B\|704\|2,048\|2.1e-03\|1176.8M\|16\|16\|69,369\|100.0B.<br>docs/paper/tables/suite\_stats.tex:18-18 | "1B\|704\|2,048\|2.1e-03\|1176.8M\|16\|16\|69,369\|100.0B" | value={"batch\_size":"704","heads":"16","hidden\_dimension":"2,048","layers":"16","learning\_rate":"2.2e-03","model\_name":"1B","model\_size":"1176.8M","tokens\_trained":"100.0B","training\_steps":"69,359"}; diagnostics=\["learning\_rate: expected 2.1e-03, observed 2.2e-03","training\_steps: expected 69,369, observed 69,359","catalog-derived evidence is below the required training-rerun boundary"\] | contradicted | required=training\_rerun; actual=paper\_or\_final\_artifact | — | method=suite\_config\_comparison; provenance=paper\_derived; policy=suite\_config\_v1; verifier=suite\_config | — | \[\] |

## Reproduced

None in the selected run.

## Source-only matches

These results are not independent reproductions. Full recorded details are in the immutable observations file identified above.

| Group | Count | Claim IDs |
| --- | ---: | --- |
| source or author-artifact agreement only | 154 | DD-0001, DD-0022, DD-0025, DD-0030, DD-0031, DD-0037, DD-0102, DD-0126, DD-0151, DD-0152<br>DD-0163, DD-0171, DD-0172, DD-0173, DD-0182, DD-0187, DD-0217, DD-0243, DD-0245, DD-0250<br>DD-0252, DD-0253, DD-0260, DD-0262, DD-0264, DD-0266, DD-0267, DD-0270, DD-0271, DD-0297<br>DD-0298, DD-0299, DD-0301, DD-0302, DD-0303, DD-0304, DD-0305, DD-0306, DD-0307, DD-0308<br>DD-0310, DD-0316, DD-0331, DD-0332, DD-0333, DD-0334, DD-0335, DD-0336, DD-0337, DD-0338<br>DD-0339, DD-0340, DD-0341, DD-0342, DD-0343, DD-0344, DD-0345, DD-0346, DD-0347, DD-0348<br>DD-0349, DD-0350, DD-0351, DD-0352, DD-0353, DD-0354, DD-0355, DD-0356, DD-0357, DD-0358<br>DD-0359, DD-0360, DD-0361, DD-0362, DD-0363, DD-0364, DD-0365, DD-0366, DD-0367, DD-0368<br>DD-0369, DD-0370, DD-0371, DD-0372, DD-0373, DD-0374, DD-0375, DD-0376, DD-0377, DD-0378<br>DD-0379, DD-0380, DD-0381, DD-0382, DD-0383, DD-0384, DD-0385, DD-0386, DD-0387, DD-0388<br>DD-0389, DD-0390, DD-0391, DD-0392, DD-0393, DD-0394, DD-0395, DD-0396, DD-0397, DD-0398<br>DD-0399, DD-0400, DD-0401, DD-0402, DD-0403, DD-0404, DD-0405, DD-0406, DD-0407, DD-0408<br>DD-0409, DD-0410, DD-0411, DD-0412, DD-0413, DD-0414, DD-0415, DD-0416, DD-0417, DD-0418<br>DD-0419, DD-0420, DD-0421, DD-0422, DD-0423, DD-0424, DD-0425, DD-0426, DD-0427, DD-0428<br>DD-0429, DD-0430, DD-0431, DD-0432, DD-0433, DD-0434, DD-0435, DD-0436, DD-0437, DD-0438<br>DD-0439, DD-0440, DD-0441, DD-0442 |

## Blocked: missing input

Claims are grouped by the stable missing input IDs and recorded blocker reason.

| Group | Count | Claim IDs |
| --- | ---: | --- |
| missing inputs=\["artifact-release-manifest"\]; reason=no pinned artifact-release manifest is available in this run | 18 | DD-0003, DD-0032, DD-0033, DD-0034, DD-0041, DD-0048, DD-0049, DD-0050, DD-0058, DD-0059<br>DD-0060, DD-0088, DD-0153, DD-0246, DD-0256, DD-0257, DD-0258, DD-0259 |
| missing inputs=\["corpus-construction-manifest"\]; reason=no corpus-construction manifest is available in this run | 12 | DD-0005, DD-0006, DD-0038, DD-0039, DD-0047, DD-0062, DD-0063, DD-0064, DD-0065, DD-0066<br>DD-0067, DD-0073 |
| missing inputs=\["evaluation-rerun-results"\]; reason=no evaluation-rerun results are available in this run | 14 | DD-0035, DD-0131, DD-0132, DD-0133, DD-0134, DD-0135, DD-0136, DD-0137, DD-0138, DD-0139<br>DD-0140, DD-0141, DD-0147, DD-0254 |
| missing inputs=\["olmes-summary:params=150M:step=38157:metric=primary\_metric"\]; reason=the exact paper-final 150M primary-metric decision summary is absent | 1 | DD-0011 |
| missing inputs=\["training-run-manifest"\]; reason=no completed training-run manifest is available in this run | 17 | DD-0007, DD-0008, DD-0009, DD-0020, DD-0089, DD-0090, DD-0092, DD-0095, DD-0099, DD-0104<br>DD-0128, DD-0129, DD-0244, DD-0247, DD-0251, DD-0274, DD-0275 |
| missing inputs=\["training-run-manifest"\]; reason=the catalog and model schedule derivations do not encode per-run seed stopping policy or completed training-run evidence | 1 | DD-0273 |
| missing inputs=\["training-run-manifest"\]; reason=the catalog defines five seed aliases but does not map three aliases to every recipe/configuration pair | 1 | DD-0272 |

## Blocked: unspecified method

Claims are grouped by unresolved method ID; recorded reasons remain visible for action.

| Group | Count | Claim IDs |
| --- | ---: | --- |
| unresolved method=benefit\_threshold\_and\_scale\_window; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0208 |
| unresolved method=change\_point\_and\_trend\_assessment; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0177, DD-0178, DD-0179 |
| unresolved method=checkpoint\_progress\_boundary; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0327 |
| unresolved method=code\_task\_metric\_comparison; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 4 | DD-0213, DD-0221, DD-0224, DD-0226 |
| unresolved method=comparable\_error\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0311, DD-0330 |
| unresolved method=compute\_equivalent\_matching; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0165 |
| unresolved method=cross\_metric\_pattern\_assessment; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0312 |
| unresolved method=crossover\_definition; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0191 |
| unresolved method=crossover\_definition\_and\_noise\_disambiguation; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0194 |
| unresolved method=downstream\_impact\_evidence; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0265 |
| unresolved method=early\_seed\_stopping\_checkpoint; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0096 |
| unresolved method=gpu\_hour\_accounting; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0261 |
| unresolved method=high\_accuracy\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0192 |
| unresolved method=impact\_comparison\_basis; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0023 |
| unresolved method=last\_checkpoint\_window\_selection; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0116 |
| unresolved method=log\_linear\_trend\_assessment; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0169 |
| unresolved method=mixture\_sampling\_policy; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0085, DD-0086, DD-0087 |
| unresolved method=most\_scales\_definition; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0199 |
| unresolved method=multiscale\_compute\_accounting; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0183 |
| unresolved method=noise\_floor\_definition; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0225 |
| unresolved method=noise\_spread\_calculation; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 9 | DD-0056, DD-0057, DD-0209, DD-0210, DD-0211, DD-0212, DD-0218, DD-0219, DD-0220 |
| unresolved method=non\_trivial\_performance\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0142 |
| unresolved method=open\_suite\_extent\_comparison; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0004 |
| unresolved method=optimality\_scale\_and\_task\_aggregation; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0198 |
| unresolved method=pairwise\_noise\_combination; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0214 |
| unresolved method=pairwise\_ranking\_and\_tie\_policy; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 9 | DD-0010, DD-0021, DD-0036, DD-0044, DD-0105, DD-0108, DD-0123, DD-0124, DD-0127 |
| unresolved method=parameter\_reparameterization; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0323 |
| unresolved method=pareto\_frontier\_construction; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 4 | DD-0013, DD-0054, DD-0180, DD-0181 |
| unresolved method=predictable\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0149, DD-0166, DD-0175 |
| unresolved method=prediction\_error\_aggregation; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0309 |
| unresolved method=prior\_work\_counterfactual\_survey; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0028 |
| unresolved method=prior\_work\_validation\_survey; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0029 |
| unresolved method=proxy\_metric\_comparison; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0055, DD-0196 |
| unresolved method=recipe\_filter\_implementation; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 10 | DD-0074, DD-0075, DD-0076, DD-0077, DD-0079, DD-0080, DD-0081, DD-0082, DD-0083, DD-0084 |
| unresolved method=recipe\_task\_scope\_for\_two\_percent\_claim; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0098 |
| unresolved method=scaling\_fit\_optimizer\_bounds\_initialization; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0114 |
| unresolved method=seed\_assignment\_and\_data\_order\_policy; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0091 |
| unresolved method=standard\_deviation\_convention; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0046, DD-0215, DD-0216 |
| unresolved method=substantially\_better\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0119 |
| unresolved method=target\_metric\_switch\_analysis; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0227 |
| unresolved method=task\_compute\_comparison; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 4 | DD-0052, DD-0053, DD-0150, DD-0167 |
| unresolved method=task\_curve\_comparison; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0148, DD-0168, DD-0174 |
| unresolved method=task\_metric\_and\_compute\_selection; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 5 | DD-0014, DD-0015, DD-0016, DD-0017, DD-0018 |
| unresolved method=task\_split\_and\_item\_selection; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0146 |
| unresolved method=tie\_handling\_for\_kendall\_equivalence; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0125 |
| unresolved method=top\_method\_tie\_and\_scope; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0189 |
| unresolved method=trend\_assessment; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0164, DD-0206 |
| unresolved method=trend\_comparison\_window; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 1 | DD-0205 |
| unresolved method=trend\_similarity\_measure; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0197, DD-0207 |
| unresolved method=trend\_type\_classification; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 3 | DD-0202, DD-0203, DD-0204 |
| unresolved method=trivial\_accuracy\_threshold; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0176, DD-0222 |
| unresolved method=typical\_overtraining\_basis; reason(s)=\["the claim registry records an unresolved claim-specific method"\] | 2 | DD-0094, DD-0249 |

## External or citation-dependent

Claims are grouped by citation keys when present, otherwise by the recorded external blocker reason.

| Group | Count | Claim IDs |
| --- | ---: | --- |
| citation keys=\["2019t5","Penedo2023TheRD"\] | 1 | DD-0072 |
| citation keys=\["2019t5","commoncrawl"\] | 1 | DD-0068 |
| citation keys=\["Dubey2024TheL3","bhagia2024establishingtaskscalinglaws","gadre2024languagemodelsscalereliably"\] | 2 | DD-0027, DD-0231 |
| citation keys=\["Kang2024AutoScaleAP","Ye2024DataML"\] | 1 | DD-0234 |
| citation keys=\["Penedo2023TheRD","commoncrawl"\] | 1 | DD-0071 |
| citation keys=\["Porian2024ResolvingDI"\] | 2 | DD-0101, DD-0268 |
| citation keys=\["austin2021program","chen2021evaluatinglargelanguagemodels"\] | 1 | DD-0223 |
| citation keys=\["benallal2024smollmcorpus"\] | 1 | DD-0070 |
| citation keys=\["bhagia2024establishingtaskscalinglaws","gadre2024languagemodelsscalereliably"\] | 1 | DD-0122 |
| citation keys=\["bhagia2024establishingtaskscalinglaws","gadre2024languagemodelsscalereliably","hoffmann2022trainingcomputeoptimallargelanguage","kaplan2020scalinglawsneurallanguage"\] | 1 | DD-0109 |
| citation keys=\["bhagia2024establishingtaskscalinglaws","groeneveld2024olmoacceleratingsciencelanguage","olmo20252olmo2furious"\] | 1 | DD-0100 |
| citation keys=\["bhagia2024establishingtaskscalinglaws"\] | 2 | DD-0110, DD-0313 |
| citation keys=\["biderman2023pythiasuiteanalyzinglarge","brandfonbrener2024losstolosspredictionscalinglaws","magnusson2024palomabenchmarkevaluatinglanguage"\] | 1 | DD-0040 |
| citation keys=\["biderman2023pythiasuiteanalyzinglarge"\] | 1 | DD-0236 |
| citation keys=\["brandfonbrener2024losstolosspredictionscalinglaws","ruan2024observational"\] | 1 | DD-0233 |
| citation keys=\["brandfonbrener2024losstolosspredictionscalinglaws"\] | 1 | DD-0238 |
| citation keys=\["choshen2024hitchhikersguidescalinglaw","hoffmann2022trainingcomputeoptimallargelanguage","kaplan2020scalinglawsneurallanguage"\] | 1 | DD-0026 |
| citation keys=\["choshen2024hitchhikersguidescalinglaw"\] | 1 | DD-0235 |
| citation keys=\["commoncrawl","li2024datacomplmsearchgenerationtraining"\] | 1 | DD-0078 |
| citation keys=\["dolma"\] | 1 | DD-0061 |
| citation keys=\["du2024understanding"\] | 1 | DD-0232 |
| citation keys=\["goyal2024scaling","muennighoff2023scaling"\] | 1 | DD-0229 |
| citation keys=\["gu2024olmesstandardlanguagemodel"\] | 1 | DD-0144 |
| citation keys=\["hoffmann2022trainingcomputeoptimallargelanguage","kaplan2020scalinglawsneurallanguage"\] | 1 | DD-0228 |
| citation keys=\["hoffmann2022trainingcomputeoptimallargelanguage"\] | 2 | DD-0093, DD-0248 |
| citation keys=\["li2024datacomplmsearchgenerationtraining"\] | 6 | DD-0024, DD-0106, DD-0239, DD-0240, DD-0241, DD-0242 |
| citation keys=\["magnusson2024palomabenchmarkevaluatinglanguage"\] | 1 | DD-0237 |
| citation keys=\["schaeffer2023emergentabilitieslargelanguage"\] | 1 | DD-0154 |
| citation keys=\["schaeffer2024why"\] | 1 | DD-0230 |
| citation keys=\["zhou2024programming"\] | 1 | DD-0069 |

## Not attempted or not applicable

Claims are grouped by verdict and recorded reason.

| Group | Count | Claim IDs |
| --- | ---: | --- |
| verdict=not\_attempted; reason=no claim-specific mapping to the available normalized evaluation facts is implemented | 62 | DD-0002, DD-0012, DD-0019, DD-0042, DD-0043, DD-0045, DD-0051, DD-0097, DD-0103, DD-0107<br>DD-0111, DD-0112, DD-0113, DD-0115, DD-0117, DD-0118, DD-0120, DD-0121, DD-0130, DD-0143<br>DD-0145, DD-0155, DD-0156, DD-0157, DD-0158, DD-0159, DD-0160, DD-0161, DD-0162, DD-0170<br>DD-0184, DD-0185, DD-0186, DD-0188, DD-0190, DD-0193, DD-0195, DD-0200, DD-0201, DD-0255<br>DD-0263, DD-0290, DD-0291, DD-0292, DD-0293, DD-0294, DD-0295, DD-0296, DD-0300, DD-0314<br>DD-0315, DD-0317, DD-0318, DD-0319, DD-0320, DD-0321, DD-0322, DD-0324, DD-0325, DD-0326<br>DD-0328, DD-0329 |
