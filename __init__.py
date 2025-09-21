__version__ = "1.10.0"
__doc__ = """
Generates a simulation of an interbank network

  You can use it interactively, but if you import it, the process will be:
    # model = Model()
    #   # step by step:
    #   model.enable_backward()
    #   model.forward() # t=0 -> t=1
    #   model.backward() : reverts the last step (when executed)
    #   # all in a loop:
    #   model.simulate_full()

@author: hector@bith.net
@date:   04/2023, 09/2025
"""