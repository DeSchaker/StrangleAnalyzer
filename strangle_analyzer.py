import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt


class StrangleAnalyzer:
    def __init__(self, symbol='SPY', lookback_days=365, dte_range=(30, 45), 
                 premium=300, distance=0.08):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.dte_range = dte_range
        self.premium = premium
        self.distance = distance
        self.data = None
        self.results = None
        self.trade_history = []
        
    def fetch_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        self.data = yf.download(self.symbol, start=start_date, end=end_date)
        return self
        
    def calculate_pnl(self, entry_price, exit_price, upper_strike, lower_strike, premium):
        if lower_strike <= exit_price <= upper_strike:
            return premium
        
        if exit_price > upper_strike:
            loss = (exit_price - upper_strike) * 100
            return premium - loss
        else:
            loss = (lower_strike - exit_price) * 100
            return premium - loss
    
    def analyze(self):
        if self.data is None:
            self.fetch_data()
            
        results = []
        self.trade_history = []
        
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Rolling_Vol'] = self.data['Returns'].rolling(window=30).std() * np.sqrt(252)
        
        for distance in range(1, 21):
            dist_decimal = distance / 100
            successes = 0
            total = 0
            
            for i in range(len(self.data) - max(self.dte_range)):
                entry_date = self.data.index[i]
                entry_price = self.data['Close'].iloc[i]
                
                upper_strike = entry_price * (1 + dist_decimal)
                lower_strike = entry_price * (1 - dist_decimal)
                
                for dte in range(self.dte_range[0], self.dte_range[1] + 1):
                    if i + dte >= len(self.data):
                        break
                        
                    exit_date = self.data.index[i + dte]
                    exit_price = self.data['Close'].iloc[i + dte]
                    success = lower_strike <= exit_price <= upper_strike
                    
                    trade_info = {
                        'Distance': distance,
                        'Entry_Date': entry_date,
                        'Exit_Date': exit_date,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Upper_Strike': upper_strike,
                        'Lower_Strike': lower_strike,
                        'Success': success,
                        'DTE': dte,
                        'Price_Move_Pct': ((exit_price - entry_price) / entry_price * 100),
                        'PnL': self.calculate_pnl(entry_price, exit_price, upper_strike, 
                                                lower_strike, self.premium)
                    }
                    
                    self.trade_history.append(trade_info)
                    
                    if success:
                        successes += 1
                    total += 1
            
            success_rate = (successes / total * 100) if total > 0 else 0
            results.append({
                'Distance': f"{distance}%",
                'Success_Rate': round(success_rate, 1),
                'Sample_Size': total,
                'Distance_Numeric': distance
            })
        
        self.results = pd.DataFrame(results)
        return self
    
    def analyze_streaks(self, distance):
        trades = [t for t in self.trade_history if t['Distance'] == distance]
        trades.sort(key=lambda x: x['Entry_Date'])
        
        if not trades:
            return {
                'Max_Win_Streak': 0,
                'Max_Loss_Streak': 0,
                'Detailed_Streaks': []
            }
        
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        streaks = []
        
        current_success = trades[0]['Success']
        streak_start = 0
        
        for i in range(1, len(trades)):
            if trades[i]['Success'] == current_success:
                current_streak += 1
            else:
                streaks.append({
                    'Type': 'Win' if current_success else 'Loss',
                    'Length': current_streak,
                    'Start_Date': trades[streak_start]['Entry_Date'],
                    'End_Date': trades[i-1]['Exit_Date']
                })
                
                if current_success:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                
                current_streak = 1
                current_success = trades[i]['Success']
                streak_start = i
        
        # Add final streak
        streaks.append({
            'Type': 'Win' if current_success else 'Loss',
            'Length': current_streak,
            'Start_Date': trades[streak_start]['Entry_Date'],
            'End_Date': trades[-1]['Exit_Date']
        })
        
        if current_success:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            'Max_Win_Streak': max_win_streak,
            'Max_Loss_Streak': max_loss_streak,
            'Detailed_Streaks': streaks
        }
    
    def optimize_stop_loss(self, distance_pct=None):
        if distance_pct is None:
            distance_pct = self.distance * 100
            
        trades = [t for t in self.trade_history if t['Distance'] == distance_pct]
        if not trades:
            return None
            
        stop_loss_levels = range(50, 201, 10)
        results = []
        
        for stop_pct in stop_loss_levels:
            stop_loss = -(self.premium * (stop_pct / 100))
            total_pnl = 0
            wins = 0
            losses = 0
            max_drawdown = 0
            current_drawdown = 0
            
            for trade in trades:
                pnl = trade['PnL']
                
                # Apply stop-loss
                if pnl < stop_loss:
                    pnl = stop_loss
                
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                    current_drawdown = 0
                else:
                    losses += 1
                    current_drawdown += abs(pnl)
                    max_drawdown = max(max_drawdown, current_drawdown)
            
            win_rate = (wins / len(trades)) * 100
            avg_win = total_pnl / len(trades)
            
            results.append({
                'Stop_Loss_Pct': stop_pct,
                'Stop_Loss_Amount': stop_loss,
                'Total_PnL': total_pnl,
                'Avg_PnL': avg_win,
                'Win_Rate': win_rate,
                'Max_Drawdown': max_drawdown,
                'Trades': len(trades),
                'Sharpe': (avg_win / (max_drawdown + 1)) if max_drawdown > 0 else float('inf')
            })
        
        return pd.DataFrame(results)
    
    def plot(self):
        if self.results is None:
            self.analyze()
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        sns.barplot(data=self.results, x='Distance', y='Success_Rate', 
                   color='royalblue', ax=ax)
        ax.set_title(f'{self.symbol} Strangle Success Rate by Distance\n'
                     f'({self.dte_range[0]}-{self.dte_range[1]} DTE, '
                     f'{self.lookback_days} days lookback)',
                     pad=20, size=14)
        ax.set_xlabel('Distance from Current Price', size=12)
        ax.set_ylabel('Success Rate (%)', size=12)
        
        for i, v in enumerate(self.results['Success_Rate']):
            ax.text(i, v + 1, f'{v}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def summary(self, distance=None):
        if self.results is None:
            self.analyze()
            
        print(f"\n{self.symbol} Strangle Analysis Summary:")
        print(f"Period: {self.lookback_days} days")
        print(f"DTE Range: {self.dte_range[0]}-{self.dte_range[1]} days")
        
        if distance is not None:
            print(f"\nDetailed Analysis for {distance}% Distance:")
            
            streak_analysis = self.analyze_streaks(distance)
            trades = [t for t in self.trade_history if t['Distance'] == distance]
            
            if trades:
                pnls = [t['PnL'] for t in trades]
                print("\nP&L Analysis:")
                print(f"Total P&L: ${sum(pnls):,.2f}")
                print(f"Average P&L per trade: ${np.mean(pnls):,.2f}")
                print(f"Best trade: ${max(pnls):,.2f}")
                print(f"Worst trade: ${min(pnls):,.2f}")
                print(f"P&L Standard Deviation: ${np.std(pnls):,.2f}")
            
            print(f"\nStreak Analysis:")
            print(f"Maximum Winning Streak: {streak_analysis['Max_Win_Streak']} trades")
            print(f"Maximum Losing Streak: {streak_analysis['Max_Loss_Streak']} trades")
            
            print("\nLongest Streaks Details:")
            sorted_streaks = sorted(
                streak_analysis['Detailed_Streaks'],
                key=lambda x: (-x['Length'], 0 if x['Type'] == 'Win' else 1)
            )
            
            for streak in sorted_streaks[:5]:
                print(f"{streak['Type']} streak of {streak['Length']} trades: "
                      f"{streak['Start_Date'].strftime('%Y-%m-%d')} to "
                      f"{streak['End_Date'].strftime('%Y-%m-%d')}")
            
            print("\nStop-Loss Optimization:")
            stop_loss_results = self.optimize_stop_loss(distance)
            if stop_loss_results is not None:
                optimal_stop = stop_loss_results.loc[stop_loss_results['Sharpe'].idxmax()]
                print(f"\nOptimal Stop-Loss Analysis:")
                print(f"Stop-Loss Level: {optimal_stop['Stop_Loss_Pct']}% of premium "
                      f"(${optimal_stop['Stop_Loss_Amount']:,.2f})")
                print(f"Expected Total P&L: ${optimal_stop['Total_PnL']:,.2f}")
                print(f"Average P&L per Trade: ${optimal_stop['Avg_PnL']:,.2f}")
                print(f"Win Rate: {optimal_stop['Win_Rate']:.1f}%")
                print(f"Max Drawdown: ${optimal_stop['Max_Drawdown']:,.2f}")
                
                plt.figure(figsize=(12, 6))
                plt.plot(stop_loss_results['Stop_Loss_Pct'], 
                        stop_loss_results['Total_PnL'], 
                        marker='o')
                plt.axvline(x=optimal_stop['Stop_Loss_Pct'], 
                          color='r', linestyle='--', 
                          label=f'Optimal Stop-Loss ({optimal_stop["Stop_Loss_Pct"]}%)')
                plt.title('Stop-Loss Optimization Results')
                plt.xlabel('Stop-Loss (% of Premium)')
                plt.ylabel('Total P&L ($)')
                plt.legend()
                plt.grid(True)
                plt.show()
            
            print("\nFailed Trades Analysis:")
            failed_trades = [t for t in self.trade_history 
                           if t['Distance'] == distance and not t['Success']]
            
            if failed_trades:
                failed_df = pd.DataFrame(failed_trades)
                print("\nWorst Failures (Largest Price Moves):")
                worst_failures = failed_df.nlargest(5, 'Price_Move_Pct')
                
                for _, trade in worst_failures.iterrows():
                    print(f"Entry: {trade['Entry_Date'].strftime('%Y-%m-%d')} "
                          f"at ${trade['Entry_Price']:.2f}")
                    print(f"Exit: {trade['Exit_Date'].strftime('%Y-%m-%d')} "
                          f"at ${trade['Exit_Price']:.2f} "
                          f"({trade['Price_Move_Pct']:+.1f}%)")
                    print(f"Strikes: ${trade['Lower_Strike']:.2f} - "
                          f"${trade['Upper_Strike']:.2f}\n")
            else:
                print("No failed trades found for this distance")
        
        else:
            print("\nSuccess Rates by Distance:")
            summary_df = self.results[['Distance', 'Success_Rate', 'Sample_Size']]
            print(summary_df.to_string(index=False))
            
            optimal = self.results.loc[self.results['Success_Rate'].idxmax()]
            print(f"\nOptimal Distance: {optimal['Distance']}")
            print(f"Maximum Success Rate: {optimal['Success_Rate']}%")
            print(f"Sample Size at Optimal: {optimal['Sample_Size']:,}")


if __name__ == "__main__":
    # Initialize analyzer with parameters
    analyzer = StrangleAnalyzer(
        symbol='SPY',
        lookback_days=4096,
        dte_range=(45, 45),
        premium=300,
        distance=0.08
    )
    
    # Run analysis
    analyzer.analyze()
    
    # Display results
    print("\n=== Overall Analysis ===")
    analyzer.summary()
    
    
      
    print("\n=== Detailed Analysis for 8% Distance ===")
    analyzer.summary(distance=6)
    
    print("\n=== Generating Plots ===")
    analyzer.plot()
    plt.show()