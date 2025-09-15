from datadec.wandb_eval.wandb_analyzer import WandBDataAnalyzer


def main():
    analyzer = WandBDataAnalyzer()
    report = analyzer.generate_comprehensive_report()
    print(report)


if __name__ == "__main__":
    main()
