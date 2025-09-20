import interbank
import numpy as np

class Model(interbank.Model):
    """
    Improved version optimized for many executions
    """
    def setup_links(self):
        if len(self.banks) <= 1:
            return
        self.maxE = max(self.banks, key=lambda k: k.E).E
        max_c = max(self.banks, key=lambda k: k.C).C
        for bank in self.banks:
            bank.p = bank.E / self.maxE
            if bank.get_lender() is not None and bank.get_lender().l > 0:
                bank.lambda_ = bank.get_lender().l / bank.E
            else:
                bank.lambda_ = 0
            # bank.lambda_ = bank.l / bank.E
            bank.incrD = 0
        max_lambda = max(self.banks, key=lambda k: k.lambda_).lambda_
        for bank in self.banks:
            bank.h = bank.lambda_ / max_lambda if max_lambda > 0 else 0
            bank.A = bank.C + bank.L + bank.R
        for bank in self.banks:
            bank.c = []
            for i in range(self.config.N):
                c = 0 if i == bank.id else (1 - self.banks[i].h) * self.banks[i].A
                bank.c.append(c)
            if self.config.psi_endogenous:
                bank.psi = bank.E / self.maxE

        # optimized part ----------
        N = self.config.N
        r_i0 = self.config.r_i0
        chi = self.config.chi
        phi = self.config.phi
        xi = self.config.xi
        psi_global = self.config.psi
        psi_endogenous = self.config.psi_endogenous

        A = np.array([bank.A for bank in self.banks])
        p = np.array([bank.p for bank in self.banks])
        psi_array = np.array([bank.psi for bank in self.banks]) if psi_endogenous else np.full(N, psi_global)
        c = np.array([bank.c for bank in self.banks])  # NxN

        # Adjust ps1 to avoid 1 and division by zero errors:
        psi_array = np.where(psi_array == 1, 0.99999999999999, psi_array)

        mask_diag = np.eye(N, dtype=bool)
        mask_invalid = (p == 0)

        psi_matrix = np.repeat(psi_array[:, np.newaxis], N, axis=1)
        denom = p * c * (1 - psi_matrix)
        num = (chi * A[:, np.newaxis] - phi * A[np.newaxis, :] - (1 - p[np.newaxis, :]) * (xi * A[np.newaxis, :] - c))

        rij = np.where(mask_diag, 0, r_i0)
        valid_mask = (~mask_diag) & (~mask_invalid[np.newaxis, :]) & (c != 0) & (denom != 0)
        rij[valid_mask] = num[valid_mask] / denom[valid_mask]
        rij[rij < 0] = r_i0

        for i, bank in enumerate(self.banks):
            bank.rij = rij[i]

        # Determine r, asset_i y asset_j
        asset_i = A.copy()
        asset_j = np.sum(A) - A
        for i, bank in enumerate(self.banks):
            bank.r = np.sum(bank.rij) / (N - 1)
            bank.asset_i = asset_i[i]
            bank.asset_j = asset_j[i] / (N - 1)

        min_r = np.min([bank.r for bank in self.banks])
        # ----------
        for bank in self.banks:
            bank.mu = self.eta * (bank.C / max_c) + (1 - self.eta) * (min_r / bank.r)
        self.config.lender_change.step_setup_links(self)
        for bank in self.banks:
            log_change_lender = self.config.lender_change.change_lender(self, bank, self.t)
            self.log.debug('links', log_change_lender)


utils = interbank.Utils()
model = Model()
if __name__ == '__main__':
    utils.run_interactive(model)
