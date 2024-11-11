# CSCE 790 Neural Networks and Their Applications Exam Report

Author: Titon Barua <baruat@email.sc.edu>


## Question 2 (Model Identification)

### Example 1

The simulation in example 1 is implemented [here](https://github.com/titonbarua/coursework--csce790/blob/main/exam/model_identification_example_1.py).

![Example 1, Plant vs learned NN, training stopped at k = 500](./ex1_fn1.pdf)

![Example 1, Plant vs learned NN after training with random input for 50000 iterations. Note that the second half of the input signal amplitude is not properly predicted by the model. This is due to input signal range being (-2, 2) while training was done on range (-1, 1)](./ex1_fn2.pdf) 

![Example 1, Plant vs learned NN after training with random input for 50000 iterations. Input signal modified to match training range](./ex1_fn2_modified_signal.pdf) 


### Example 3

The simulation in example 3 is implemented [here](https://github.com/titonbarua/coursework--csce790/blob/main/exam/model_identification_example_3.py).

![Example 3, Plot of functions $f$ and $g$ with respect to neural networks](./ex3_graph_f_and_g.pdf)

![Example 3, Plot of plant output with respect to the identified model](./ex3_graph_y.pdf)

\pagebreak

## Question 3 (Model Identification and Dynamic Control)

The simulation in example 7 is implemented [here](https://github.com/titonbarua/coursework--csce790/blob/main/exam/adaptive_control_example_7.py). I have only done the first stage.

![Example 7, Model identification using technique described in example 2](./ex7_model_identification.pdf)

![Example 7, Refernce model output compared to no-control plant](./ex7_comparison_ref_model_vs_plant_no_control.pdf)

![Example 7, Refernce model output compared to plant with adaptive control](./ex7_comparison_ref_model_vs_plant_with_control.pdf)
