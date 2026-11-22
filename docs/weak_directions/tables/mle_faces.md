MLE outcomes < 1e-3 whose truth is > 0 (a face the truth is not on)

|       n |   runs |   mle_on_interior_face |   weak4_share |
|--------:|-------:|-----------------------:|--------------:|
|  128.00 |  27.00 |                  26.00 |          0.35 |
|  256.00 |  27.00 |                  20.00 |          0.35 |
|  512.00 |  27.00 |                  15.00 |          0.43 |
| 1024.00 |  27.00 |                  12.00 |          0.48 |
| 2048.00 |  12.00 |                   0.00 |        nan    |
| 4096.00 |  12.00 |                   0.00 |        nan    |

Outcomes hit:

| face_outcomes   |   count |
|:----------------|--------:|
| A.to_B          |      20 |
| B.div_AB        |      19 |
| B.death         |      19 |
| B.div_BB        |      15 |
| A.death         |      14 |
| B.div_AA        |      14 |
| A.div_AA        |      12 |
| B.to_A          |      11 |
| A.persist       |      10 |
| A.div_AB        |       9 |
| B.persist       |       9 |
| A.div_BB        |       3 |

| truth    |    n |   seed |   n_face_outcomes | face_outcomes                                          |   weak4_share_of_face_outcomes |
|:---------|-----:|-------:|------------------:|:-------------------------------------------------------|-------------------------------:|
| T1_ref   |  128 |      0 |                 1 | A.persist                                              |                           0.11 |
| T1_ref   |  128 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   |  128 |      2 |                 1 | A.death                                                |                           0.20 |
| T1_ref   |  256 |      0 |                 1 | A.persist                                              |                           0.11 |
| T1_ref   |  256 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   |  256 |      2 |                 0 |                                                        |                         nan    |
| T1_ref   |  512 |      0 |                 0 |                                                        |                         nan    |
| T1_ref   |  512 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   |  512 |      2 |                 0 |                                                        |                         nan    |
| T1_ref   | 1024 |      0 |                 0 |                                                        |                         nan    |
| T1_ref   | 1024 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   | 1024 |      2 |                 0 |                                                        |                         nan    |
| T1_ref   | 2048 |      0 |                 0 |                                                        |                         nan    |
| T1_ref   | 2048 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   | 2048 |      2 |                 0 |                                                        |                         nan    |
| T1_ref   | 4096 |      0 |                 0 |                                                        |                         nan    |
| T1_ref   | 4096 |      1 |                 0 |                                                        |                         nan    |
| T1_ref   | 4096 |      2 |                 0 |                                                        |                         nan    |
| T2_toA05 |  128 |      0 |                 1 | A.to_B                                                 |                           0.06 |
| T2_toA05 |  128 |      1 |                 1 | A.to_B                                                 |                           0.06 |
| T2_toA05 |  128 |      2 |                 1 | A.div_AB                                               |                           0.12 |
| T2_toA05 |  256 |      0 |                 0 |                                                        |                         nan    |
| T2_toA05 |  256 |      1 |                 1 | A.to_B                                                 |                           0.06 |
| T2_toA05 |  256 |      2 |                 1 | A.div_AB                                               |                           0.12 |
| T2_toA05 |  512 |      0 |                 0 |                                                        |                         nan    |
| T2_toA05 |  512 |      1 |                 0 |                                                        |                         nan    |
| T2_toA05 |  512 |      2 |                 0 |                                                        |                         nan    |
| T2_toA05 | 1024 |      0 |                 0 |                                                        |                         nan    |
| T2_toA05 | 1024 |      1 |                 0 |                                                        |                         nan    |
| T2_toA05 | 1024 |      2 |                 0 |                                                        |                         nan    |
| T2_toA05 | 2048 |      0 |                 0 |                                                        |                         nan    |
| T2_toA05 | 2048 |      1 |                 0 |                                                        |                         nan    |
| T2_toA05 | 2048 |      2 |                 0 |                                                        |                         nan    |
| T2_toA05 | 4096 |      0 |                 0 |                                                        |                         nan    |
| T2_toA05 | 4096 |      1 |                 0 |                                                        |                         nan    |
| T2_toA05 | 4096 |      2 |                 0 |                                                        |                         nan    |
| T3_Bdiv  |  128 |      0 |                 2 | A.to_B B.div_AB                                        |                           0.36 |
| T3_Bdiv  |  128 |      1 |                 3 | A.death B.div_AA B.div_AB                              |                           0.52 |
| T3_Bdiv  |  128 |      2 |                 1 | B.div_AB                                               |                           0.66 |
| T3_Bdiv  |  256 |      0 |                 1 | A.death                                                |                           0.16 |
| T3_Bdiv  |  256 |      1 |                 1 | B.div_AB                                               |                           0.66 |
| T3_Bdiv  |  256 |      2 |                 3 | A.to_B A.div_AB B.death                                |                           0.15 |
| T3_Bdiv  |  512 |      0 |                 2 | A.death B.div_AB                                       |                           0.41 |
| T3_Bdiv  |  512 |      1 |                 1 | B.div_AB                                               |                           0.66 |
| T3_Bdiv  |  512 |      2 |                 3 | A.persist B.death B.div_AB                             |                           0.34 |
| T3_Bdiv  | 1024 |      0 |                 1 | B.div_AA                                               |                           0.74 |
| T3_Bdiv  | 1024 |      1 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 1024 |      2 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 2048 |      0 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 2048 |      1 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 2048 |      2 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 4096 |      0 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 4096 |      1 |                 0 |                                                        |                         nan    |
| T3_Bdiv  | 4096 |      2 |                 0 |                                                        |                         nan    |
| T4_slowA |  128 |      0 |                 1 | A.persist                                              |                           0.17 |
| T4_slowA |  128 |      1 |                 1 | B.death                                                |                           0.66 |
| T4_slowA |  128 |      2 |                 1 | A.death                                                |                           0.33 |
| T4_slowA |  256 |      0 |                 0 |                                                        |                         nan    |
| T4_slowA |  256 |      1 |                 2 | A.persist B.death                                      |                           0.42 |
| T4_slowA |  256 |      2 |                 0 |                                                        |                         nan    |
| T4_slowA |  512 |      0 |                 0 |                                                        |                         nan    |
| T4_slowA |  512 |      1 |                 0 |                                                        |                         nan    |
| T4_slowA |  512 |      2 |                 0 |                                                        |                         nan    |
| T4_slowA | 1024 |      0 |                 0 |                                                        |                         nan    |
| T4_slowA | 1024 |      1 |                 0 |                                                        |                         nan    |
| T4_slowA | 1024 |      2 |                 0 |                                                        |                         nan    |
| T4_slowA | 2048 |      0 |                 0 |                                                        |                         nan    |
| T4_slowA | 2048 |      1 |                 0 |                                                        |                         nan    |
| T4_slowA | 2048 |      2 |                 0 |                                                        |                         nan    |
| T4_slowA | 4096 |      0 |                 0 |                                                        |                         nan    |
| T4_slowA | 4096 |      1 |                 0 |                                                        |                         nan    |
| T4_slowA | 4096 |      2 |                 0 |                                                        |                         nan    |
| P1       |  128 |      0 |                 3 | B.death B.to_A B.div_BB                                |                           0.43 |
| P1       |  128 |      1 |                 4 | A.to_B A.div_AA B.to_A B.div_AB                        |                           0.36 |
| P1       |  128 |      2 |                 4 | A.div_AA A.div_BB B.to_A B.div_AB                      |                           0.35 |
| P1       |  256 |      0 |                 3 | B.to_A B.persist B.div_AA                              |                           0.34 |
| P1       |  256 |      1 |                 3 | A.to_B A.div_AA B.div_AB                               |                           0.39 |
| P1       |  256 |      2 |                 1 | B.div_AB                                               |                           0.47 |
| P1       |  512 |      0 |                 1 | B.to_A                                                 |                           0.26 |
| P1       |  512 |      1 |                 2 | A.to_B A.div_AA                                        |                           0.35 |
| P1       |  512 |      2 |                 3 | A.div_AA B.to_A B.div_AB                               |                           0.35 |
| P1       | 1024 |      0 |                 1 | A.div_AA                                               |                           0.32 |
| P1       | 1024 |      1 |                 1 | B.div_BB                                               |                           0.51 |
| P1       | 1024 |      2 |                 2 | A.div_AA B.to_A                                        |                           0.29 |
| P2       |  128 |      0 |                 2 | A.to_B A.div_AB                                        |                           0.22 |
| P2       |  128 |      1 |                 2 | A.death B.div_AB                                       |                           0.45 |
| P2       |  128 |      2 |                 2 | A.div_AB B.persist                                     |                           0.31 |
| P2       |  256 |      0 |                 2 | A.to_B A.div_AB                                        |                           0.22 |
| P2       |  256 |      1 |                 2 | A.to_B B.death                                         |                           0.34 |
| P2       |  256 |      2 |                 3 | A.to_B B.death B.div_AB                                |                           0.40 |
| P2       |  512 |      0 |                 1 | A.div_AB                                               |                           0.35 |
| P2       |  512 |      1 |                 2 | A.div_AB B.death                                       |                           0.47 |
| P2       |  512 |      2 |                 3 | A.to_B B.death B.div_AB                                |                           0.40 |
| P2       | 1024 |      0 |                 2 | A.div_AB B.death                                       |                           0.47 |
| P2       | 1024 |      1 |                 1 | B.death                                                |                           0.58 |
| P2       | 1024 |      2 |                 2 | B.death B.div_AB                                       |                           0.56 |
| P3       |  128 |      0 |                 4 | A.to_B A.div_AA B.death B.persist                      |                           0.41 |
| P3       |  128 |      1 |                 5 | A.death A.to_B A.div_AA B.persist B.div_AB             |                           0.39 |
| P3       |  128 |      2 |                 4 | A.death A.to_B A.div_AA B.div_BB                       |                           0.41 |
| P3       |  256 |      0 |                 2 | A.to_B B.div_AA                                        |                           0.30 |
| P3       |  256 |      1 |                 5 | A.death A.to_B A.div_AA B.persist B.div_AB             |                           0.39 |
| P3       |  256 |      2 |                 2 | B.death B.div_BB                                       |                           0.49 |
| P3       |  512 |      0 |                 2 | A.death A.div_AA                                       |                           0.45 |
| P3       |  512 |      1 |                 2 | A.to_B B.death                                         |                           0.37 |
| P3       |  512 |      2 |                 3 | A.persist B.persist B.div_AA                           |                           0.32 |
| P3       | 1024 |      0 |                 1 | A.death                                                |                           0.37 |
| P3       | 1024 |      1 |                 0 |                                                        |                         nan    |
| P3       | 1024 |      2 |                 3 | A.death A.to_B B.div_BB                                |                           0.37 |
| P4       |  128 |      0 |                 6 | A.death A.persist A.div_BB B.persist B.div_AA B.div_BB |                           0.36 |
| P4       |  128 |      1 |                 3 | B.persist B.div_AA B.div_BB                            |                           0.57 |
| P4       |  128 |      2 |                 3 | B.death B.to_A B.div_BB                                |                           0.63 |
| P4       |  256 |      0 |                 3 | B.death B.to_A B.div_BB                                |                           0.63 |
| P4       |  256 |      1 |                 2 | B.persist B.div_AA                                     |                           0.61 |
| P4       |  256 |      2 |                 3 | A.persist B.div_AA B.div_BB                            |                           0.48 |
| P4       |  512 |      0 |                 3 | B.death B.to_A B.div_BB                                |                           0.63 |
| P4       |  512 |      1 |                 3 | B.to_A B.div_AB B.div_BB                               |                           0.61 |
| P4       |  512 |      2 |                 3 | A.persist B.div_AA B.div_BB                            |                           0.48 |
| P4       | 1024 |      0 |                 2 | B.death B.div_BB                                       |                           0.65 |
| P4       | 1024 |      1 |                 1 | B.div_BB                                               |                           0.50 |
| P4       | 1024 |      2 |                 2 | A.div_BB B.div_AA                                      |                           0.43 |
| T5_C7    |  128 |      0 |                 1 | A.death                                                |                           0.52 |
| T5_C7    |  128 |      1 |                 1 | B.div_AA                                               |                           0.19 |
| T5_C7    |  128 |      2 |                 2 | A.persist B.div_AA                                     |                           0.36 |
| T5_C7    |  256 |      0 |                 0 |                                                        |                         nan    |
| T5_C7    |  256 |      1 |                 0 |                                                        |                         nan    |
| T5_C7    |  256 |      2 |                 1 | B.div_AA                                               |                           0.19 |
| T5_C7    |  512 |      0 |                 0 |                                                        |                         nan    |
| T5_C7    |  512 |      1 |                 0 |                                                        |                         nan    |
| T5_C7    |  512 |      2 |                 0 |                                                        |                         nan    |
| T5_C7    | 1024 |      0 |                 0 |                                                        |                         nan    |
| T5_C7    | 1024 |      1 |                 0 |                                                        |                         nan    |
| T5_C7    | 1024 |      2 |                 0 |                                                        |                         nan    |
