import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


class ExtremeCorrection:

    def __init__(self, data: pd.DataFrame, data_var: str = 'Hs', frequency: float = 365.25, 
                 year_var: str = 'yyyy', month_var: str = 'mm'
                 ):
        """
        Inicializa el algoritmo de corrección de extremos.

        Args:
            data (pd.DataFrame): Datos observados incluyendo variables mes ('mm') y año ('yyyy').
            data_var (str): Nombre de la variable que se quiere corregir.
            frequency (float): Número de datos al año (por defecto 365.25 para datos diarios)
        """

        self.data = data
        self.data_var = data_var
        self.month_var = month_var
        self.year_var = year_var
        self.frequency = frequency

        # Definir variables
        self.max_data = None
        self.max_idx = None
        self.max_data_sorted = None
        self.max_idx_sorted = None
        self.pit_data = self.data[self.data_var].values
        self.pit_data_sorted = np.sort(self.pit_data)

        self.obtain_data()

        # Parámetros asociados a la GEV estacionaria (localización, escala, forma)
        self.gev_parameters = None

        # Datos corregidos
        self.max_data_corrected = None
        self.pit_data_corrected = None

        self.gev_fit()
        self.correction()

    def obtain_data(self):

        self.max_data = self.data.groupby([self.year_var], as_index=False)[self.data_var].max()[self.data_var].values
        self.max_idx = self.data.groupby([self.year_var])[self.data_var].idxmax().values
        self.max_data_sorted = np.sort(self.max_data)
        self.max_idx_sorted = np.argsort(self.max_data)

    def ecdf(self, x):
        return np.arange(1,len(x)+1)/(len(x)+1)

    def rp_year(self, probs):
        return 1/(1-probs)
    
    def rescaled_rp(self, probs):
        return 1/(1-probs)/self.frequency
    
    def return_period_plot(self):
        """
        Dibujar el gráfico de periodos de retorno sin corrección
        """
        fig = plt.figure(figsize=(10,6))
        ax= fig.add_subplot()
        ax.semilogx(self.rescaled_rp(self.ecdf(self.pit_data)), self.pit_data_sorted, linewidth=0, marker='o',markersize=1.5, label=f'Daily Data ({self.data_var})')
        ax.semilogx(self.rp_year(self.ecdf(self.max_data)), self.max_data_sorted, color = 'orange',linewidth=0, marker='o',markersize=1.5, label=f'Annual Maxima ({self.data_var})')

        # Configurar escala logarítmica y ticks personalizados
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Mantiene los números sin notación científica

        # Etiquetas y grid
        ax.set_xlabel('Return Period (Years)')
        ax.set_ylabel(f'{self.data_var}')
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Mostrar gráfico
        plt.legend()
        plt.show()

    def gev_fit(self, qqplot=False):
        """
        Ajustar una GEV estacionaria al régimen extremal
        """
        shape_gev, loc_gev, scale_gev = stats.genextreme.fit(self.max_data) 

        self.gev_parameters = [loc_gev, scale_gev, shape_gev]

        if qqplot:
            sm.qqplot(self.max_data, stats.genextreme, distargs=(shape_gev,), loc=loc_gev, scale=scale_gev, line="45")
            plt.title("Q-Q Plot for Normal Distribution Fit")
            plt.grid()
            plt.show()

    def qgev(self,probs):
        return stats.genextreme.ppf(probs, self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
    
    def correction(self):
        self.max_data_corrected = self.qgev(self.ecdf(self.max_data))

        self.pit_data_corrected = self.pit_data.copy()
        for block_idx, max_value in enumerate(self.max_data_sorted):
            if block_idx >= len(self.max_data)-1:
                self.pit_data_corrected[self.pit_data >= self.max_data_sorted[-1]] = self.max_data_corrected[-1]
                break
            else:
                for idx, daily_value in enumerate(self.pit_data):
                    if daily_value >= self.max_data_sorted[block_idx] and daily_value < self.max_data_sorted[block_idx+1]:
                        self.pit_data_corrected[idx] = self.max_data_corrected[block_idx]+((daily_value-self.max_data_sorted[block_idx])/(self.max_data_sorted[block_idx+1]-self.max_data_sorted[block_idx]))*(self.max_data_corrected[block_idx+1]-self.max_data_corrected[block_idx])

    def plot_ts(self, savefig=False):
        plt.figure(figsize=(20,6))

        plt.plot(np.arange(1,len(self.pit_data)+1)/self.frequency, self.pit_data, label='Daily Data', alpha=0.8)
        plt.plot(np.arange(1,len(self.pit_data)+1)/self.frequency, self.pit_data_corrected, label='Corrected Daily Data', alpha=0.8)
        plt.scatter(np.arange(1,len(self.pit_data)+1)[self.max_idx]/self.frequency, self.max_data, label='Annual Maxima')
        plt.scatter(np.arange(1,len(self.pit_data)+1)[self.max_idx]/self.frequency, self.pit_data_corrected[self.max_idx], label='Corrected Annual Maxima',)
        plt.legend()
        plt.xticks(np.arange(1,len(self.max_data)+1))
        plt.grid()
        if savefig:
            plt.savefig("Figuras/time_series.png", dpi=200)
        plt.show()

    def return_period_plot_corrected(self, without_corr=False, savefig=False):

        x_values_gev = np.linspace(self.max_data_sorted[0], self.max_data_sorted[-1]+0.2, 1000)
        return_perd_values_gev = 1/(1-stats.genextreme.cdf(x_values_gev,self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1]))

        ecdf_ev_probs_corrected = stats.genextreme.cdf(self.qgev(self.ecdf(self.max_data)), self.gev_parameters[2], loc=self.gev_parameters[0], scale=self.gev_parameters[1])
        T_ev_corrected = 1/(1-ecdf_ev_probs_corrected)

        ecdf_pt_probs_corrected = np.arange(1,len(self.pit_data_corrected)+1)/(len(self.pit_data_corrected)+1)
        T_pt_corrected = 1/(1-ecdf_pt_probs_corrected)/self.frequency

        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot()
        #T_pt, np.sort(hspt)
        #ax.semilogx(T_pt, np.sort(hspt), linewidth=1,markersize=1.5)
        ax.semilogx(return_perd_values_gev, np.sort(x_values_gev), color = 'orange',linestyle='dashed', label='Fitted GEV')

        # Datos corregidos
        ax.semilogx(T_pt_corrected, np.sort(self.pit_data_corrected), linewidth=0, marker='o',markersize=3, label=f'Corrected Daily Data (${self.data_var}$)')
        ax.semilogx(T_ev_corrected, self.qgev(self.ecdf(self.max_data)), color = 'orange',linewidth=0, marker='o',markersize=3, label=fr'Corrected Annual Maxima (${self.data_var}^{{max}}$)')

        # Sin corregir
        if without_corr:
            ax.semilogx(self.rescaled_rp(self.ecdf(self.pit_data)), self.pit_data_sorted, linewidth=0, color='purple', marker='o',markersize=1.5, label=f'Daily Data (${self.data_var}$)', alpha=0.6)
            ax.semilogx(self.rp_year(self.ecdf(self.max_data)), self.max_data_sorted, color = 'green',linewidth=0, marker='o',markersize=1.5, label=fr'Annual Maxima (${self.data_var}^{{max}}$)', alpha=0.6)


        # Configurar escala logarítmica y ticks personalizados
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 250, 1000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Mantiene los números sin notación científica

        # Etiquetas y grid
        ax.set_xlabel('Return Period (Years)')
        ax.set_ylabel(f'{self.data_var}')
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Mostrar gráfico
        plt.legend()
        if savefig:
            if without_corr:
                name = "non_and_corrected_return_periods"
            else:
                name = "corrected_return_periods"
            plt.savefig(f"Figuras/{name}.png", dpi=200)
        plt.show()

        return fig