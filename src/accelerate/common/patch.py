import types
import torch
import typing as T


T_Module  = T.TypeVar("T_Module", bound=torch.nn.Module)
T_FwdArgs = T.ParamSpec("T_FwdArgs")
T_FwdOuts = T.TypeVar("T_FwdOuts")
T_Args = T.ParamSpec("T_Args")
T_Return = T.TypeVar("T_Return")


def patch_torch_forward(
    module: T_Module,
    accelerate_fwd_factory: T.Callable[[T.Callable[T_FwdArgs, T_FwdOuts]], T.Callable[T.Concatenate[T_Module, T_FwdArgs], T_FwdOuts]]
) -> T_Module:
    original_forward: T.Callable[T_FwdArgs, T_FwdOuts] = module.forward
    
    accelerate_forward = accelerate_fwd_factory(original_forward)
    module.forward = types.MethodType(accelerate_forward, module)
    
    return module


def patch_torch_method(
    module: T_Module,
    method_name: str,
    method_factory: T.Callable[[T.Callable[T_Args, T_Return]], T.Callable[T.Concatenate[T_Module, T_Args], T_Return]]
) -> T_Module:
    """
    Generalized method to monkey patch any method on a PyTorch module.
    
    Args:
        module: The PyTorch module to patch
        method_name: Name of the method to replace (e.g., "forward", "backward", etc.)
        method_factory: Factory function that takes the original method and returns the new method
        
    Returns:
        The patched module
        
    Raises:
        AttributeError: If the method doesn't exist on the module
    """
    if not hasattr(module, method_name):
        raise AttributeError(f"Module {type(module).__name__} has no method '{method_name}'")
    
    original_method: T.Callable[T_Args, T_Return] = getattr(module, method_name)
    
    # Create the new method using the factory
    new_method = method_factory(original_method)
    
    # Bind the new method to the module
    setattr(module, method_name, types.MethodType(new_method, module))
    
    return module
