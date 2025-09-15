# src/trading_system/models/pinn_loader.py
import torch
import importlib.util
import logging

logger = logging.getLogger(__name__)


class PINNWrapper:
    def __init__(self, model_class, weights_path: str, device: str = None):
        # Seleciona device (cuda se disponível, senão cpu)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializa modelo no device correto
        self.model = model_class().to(self.device)

        # Carrega pesos no mesmo device
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, inputs):
        """Recebe np.array ou lista, move para o device do modelo e retorna no CPU."""
        with torch.no_grad():
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            if x.ndim == 1:  # caso venha um único vetor
                x = x.unsqueeze(0)

            preds = self.model(x).squeeze()

            # Retorna sempre para CPU como numpy
            return preds.detach().cpu().numpy()


def load_pinn(model_path: str, weights_path: str, device: str = None):
    """
    Carrega modelo PINN dinamicamente a partir do código + pesos.
    Garante que modelo e tensores fiquem no mesmo device.
    """
    try:
        spec = importlib.util.spec_from_file_location("PINN_model", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "EuropeanCallPINN"):
            model_class = module.EuropeanCallPINN
        else:
            raise ImportError("EuropeanCallPINN não encontrado no arquivo PINN_model.py")

        wrapper = PINNWrapper(model_class, weights_path, device)
        logger.info(f"[Hermes] PINN carregado com sucesso no device: {wrapper.device}")
        return wrapper

    except Exception as e:
        logger.error(f"[Hermes] Falha ao carregar PINN: {e}", exc_info=True)
        raise
