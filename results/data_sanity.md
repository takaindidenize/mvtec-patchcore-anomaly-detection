# Data sanity report

Data root: `data/mvtec`
Categories audited: 15

## bottle

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 209 | - |
| `test/broken_large` | 20 | 20 |
| `test/broken_small` | 22 | 22 |
| `test/contamination` | 21 | 21 |
| `test/good` | 20 | - |

- Image sizes: 900x900 (83)
- Mask value sets: [0, 255]
- Findings: 0

## cable

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 224 | - |
| `test/bent_wire` | 13 | 13 |
| `test/cable_swap` | 12 | 12 |
| `test/combined` | 11 | 11 |
| `test/cut_inner_insulation` | 14 | 14 |
| `test/cut_outer_insulation` | 10 | 10 |
| `test/good` | 58 | - |
| `test/missing_cable` | 12 | 12 |
| `test/missing_wire` | 10 | 10 |
| `test/poke_insulation` | 10 | 10 |

- Image sizes: 1024x1024 (150)
- Mask value sets: [0, 255]
- Findings: 0

## capsule

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 219 | - |
| `test/crack` | 23 | 23 |
| `test/faulty_imprint` | 22 | 22 |
| `test/good` | 23 | - |
| `test/poke` | 21 | 21 |
| `test/scratch` | 23 | 23 |
| `test/squeeze` | 20 | 20 |

- Image sizes: 1000x1000 (132)
- Mask value sets: [0, 255]
- Findings: 0

## carpet

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 280 | - |
| `test/color` | 19 | 19 |
| `test/cut` | 17 | 17 |
| `test/good` | 28 | - |
| `test/hole` | 17 | 17 |
| `test/metal_contamination` | 17 | 17 |
| `test/thread` | 19 | 19 |

- Image sizes: 1024x1024 (117)
- Mask value sets: [0, 255]
- Findings: 0

## grid

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 264 | - |
| `test/bent` | 12 | 12 |
| `test/broken` | 12 | 12 |
| `test/glue` | 11 | 11 |
| `test/good` | 21 | - |
| `test/metal_contamination` | 11 | 11 |
| `test/thread` | 11 | 11 |

- Image sizes: 1024x1024 (78)
- Mask value sets: [0, 255]
- Findings: 0

## hazelnut

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 391 | - |
| `test/crack` | 18 | 18 |
| `test/cut` | 17 | 17 |
| `test/good` | 40 | - |
| `test/hole` | 18 | 18 |
| `test/print` | 17 | 17 |

- Image sizes: 1024x1024 (110)
- Mask value sets: [0, 255]
- Findings: 0

## leather

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 245 | - |
| `test/color` | 19 | 19 |
| `test/cut` | 19 | 19 |
| `test/fold` | 17 | 17 |
| `test/glue` | 19 | 19 |
| `test/good` | 32 | - |
| `test/poke` | 18 | 18 |

- Image sizes: 1024x1024 (124)
- Mask value sets: [0, 255]
- Findings: 0

## metal_nut

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 220 | - |
| `test/bent` | 25 | 25 |
| `test/color` | 22 | 22 |
| `test/flip` | 23 | 23 |
| `test/good` | 22 | - |
| `test/scratch` | 23 | 23 |

- Image sizes: 700x700 (115)
- Mask value sets: [0, 255]
- Findings: 0

## pill

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 267 | - |
| `test/color` | 25 | 25 |
| `test/combined` | 17 | 17 |
| `test/contamination` | 21 | 21 |
| `test/crack` | 26 | 26 |
| `test/faulty_imprint` | 19 | 19 |
| `test/good` | 26 | - |
| `test/pill_type` | 9 | 9 |
| `test/scratch` | 24 | 24 |

- Image sizes: 800x800 (167)
- Mask value sets: [0, 255]
- Findings: 0

## screw

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 320 | - |
| `test/good` | 41 | - |
| `test/manipulated_front` | 24 | 24 |
| `test/scratch_head` | 24 | 24 |
| `test/scratch_neck` | 25 | 25 |
| `test/thread_side` | 23 | 23 |
| `test/thread_top` | 23 | 23 |

- Image sizes: 1024x1024 (160)
- Mask value sets: [0, 255]
- Findings: 0

## tile

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 230 | - |
| `test/crack` | 17 | 17 |
| `test/glue_strip` | 18 | 18 |
| `test/good` | 33 | - |
| `test/gray_stroke` | 16 | 16 |
| `test/oil` | 18 | 18 |
| `test/rough` | 15 | 15 |

- Image sizes: 840x840 (117)
- Mask value sets: [0, 255]
- Findings: 0

## toothbrush

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 60 | - |
| `test/defective` | 30 | 30 |
| `test/good` | 12 | - |

- Image sizes: 1024x1024 (42)
- Mask value sets: [0, 255]
- Findings: 0

## transistor

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 213 | - |
| `test/bent_lead` | 10 | 10 |
| `test/cut_lead` | 10 | 10 |
| `test/damaged_case` | 10 | 10 |
| `test/good` | 60 | - |
| `test/misplaced` | 10 | 10 |

- Image sizes: 1024x1024 (100)
- Mask value sets: [0, 255]
- Findings: 0

## wood

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 247 | - |
| `test/color` | 8 | 8 |
| `test/combined` | 11 | 11 |
| `test/good` | 19 | - |
| `test/hole` | 10 | 10 |
| `test/liquid` | 10 | 10 |
| `test/scratch` | 21 | 21 |

- Image sizes: 1024x1024 (79)
- Mask value sets: [0, 255]
- Findings: 0

## zipper

| Split | Images | Masks |
|---|---:|---:|
| `train/good` | 240 | - |
| `test/broken_teeth` | 19 | 19 |
| `test/combined` | 16 | 16 |
| `test/fabric_border` | 17 | 17 |
| `test/fabric_interior` | 16 | 16 |
| `test/good` | 32 | - |
| `test/rough` | 17 | 17 |
| `test/split_teeth` | 18 | 18 |
| `test/squeezed_teeth` | 16 | 16 |

- Image sizes: 1024x1024 (151)
- Mask value sets: [0, 255]
- Findings: 0

## Findings

None.
