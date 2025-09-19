# src/trading_system/models/pinn_architecture.py
import torch

class EuropeanCallPINN(torch.nn.Module):
    """
    Define a arquitetura da Rede Neural para o PINN de precificação de opções europeias.
    Combina uma rede rasa e uma profunda com Fourier Features.
    """
    def __init__(self, S_min: float, S_max: float, K_min: float, K_max: float, T_max: float, r: float):
        super().__init__()
        
        # Camada de Fourier Features para mapear as entradas para um espaço de maior dimensão
        self.fourier = torch.nn.Linear(6, 128)
        torch.nn.init.normal_(self.fourier.weight, mean=0, std=10.0)

        # Componente Raso (Shallow) da rede
        self.shallow = torch.nn.Sequential(
            torch.nn.Linear(256, 64), 
            torch.nn.SiLU()
        )

        # Componente Profundo (Deep) da rede
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(256, 512), torch.nn.SiLU(),
            torch.nn.Linear(512, 512), torch.nn.SiLU(),
            torch.nn.Linear(512, 512), torch.nn.SiLU(),
            torch.nn.Linear(512, 512), torch.nn.SiLU()
        )

        # Camada final que combina as saídas das redes rasa e profunda
        self.combiner = torch.nn.Linear(64 + 512, 1)

        # Registra os parâmetros financeiros como buffers para que sejam movidos para a GPU com .to(device)
        self.register_buffer('S_min', torch.tensor(S_min, dtype=torch.float32))
        self.register_buffer('S_max', torch.tensor(S_max, dtype=torch.float32))
        self.register_buffer('K_min', torch.tensor(K_min, dtype=torch.float32))
        self.register_buffer('K_max', torch.tensor(K_max, dtype=torch.float32))
        self.register_buffer('T_max', torch.tensor(T_max, dtype=torch.float32))
        self.register_buffer('r_rate', torch.tensor(r, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Desnormaliza as entradas para aplicar as restrições financeiras
        S = x[:, 0:1] * (self.S_max - self.S_min) + self.S_min
        t = x[:, 1:2] * self.T_max
        K = x[:, 2:3] * (self.K_max - self.K_min) + self.K_min
        
        # Mapeamento de Fourier: concatena seno e cosseno
        x_fourier = torch.cat([
            torch.sin(self.fourier(x)),
            torch.cos(self.fourier(x))
        ], dim=1)

        # Passa pelas duas redes
        shallow_out = self.shallow(x_fourier)
        deep_out = self.deep(x_fourier)
        
        # Combina as saídas
        combined = torch.cat([shallow_out, deep_out], dim=1)
        C_raw = self.combiner(combined)
        
        # Aplicação de Hard Constraints (Restrições Fortes)
        # 1. O preço da opção não pode ser menor que o valor intrínseco max(S-K, 0)
        intrinsic_value = torch.relu(S - K)
        
        # 2. Fator de decaimento temporal que garante que o preço convirja para o valor intrínseco no vencimento
        time_decay_factor = 1 - torch.exp(-self.r_rate * t)
        
        # A constante A e a função sigmoid garantem que o prêmio de risco seja sempre positivo
        A = 2.0 
        
        # O preço final é o valor intrínseco + um prêmio de risco que decai com o tempo
        return intrinsic_value + time_decay_factor * (S * A * torch.sigmoid(C_raw))