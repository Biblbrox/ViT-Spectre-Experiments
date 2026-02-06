import polars as pl


class ProfilerParser:
    def __init__(self, prof):
        """
        prof : torch.profiler.profile object (already executed)
        """
        self.prof = prof
        self.df = self._parse_events()

    def _parse_events(self) -> pl.DataFrame:
        events = self.prof.key_averages()
        rows = []

        for evt in events:
            rows.append({
                "name": evt.key,
                "calls": evt.count,
                "cpu_total_ms": evt.cpu_time_total / 1000,
                "cpu_self_ms": evt.self_cpu_time_total / 1000,
                "cuda_total_ms": evt.device_time_total / 1000,
                "cuda_self_ms": evt.self_device_time_total / 1000,
                "cpu_mem_mb": evt.cpu_memory_usage / (1024**2),
                "cuda_mem_mb": evt.device_memory_usage / (1024**2),
            })

        return pl.DataFrame(rows)

    def remove_idle(self):
        """Remove entries with zero CUDA time."""
        self.df = self.df.filter(pl.col("cuda_total_ms") > 0)
        return self

    def filter_name(self, pattern: str):
        """Keep only rows whose name matches regex."""
        self.df = self.df.filter(pl.col("name").str.contains(pattern))
        return self

    def add_percentages(self):
        total = self.df["cuda_self_ms"].sum()
        if total > 0:
            self.df = self.df.with_columns((pl.col("cuda_self_ms") / total * 100).alias("cuda_%"))
        return self

    def round(self, digits=3):
        self.df = self.df.with_columns([
            pl.col("cpu_total_ms").round(digits),
            pl.col("cpu_self_ms").round(digits),
            pl.col("cuda_total_ms").round(digits),
            pl.col("cuda_self_ms").round(digits),
            pl.col("cpu_mem_mb").round(2),
            pl.col("cuda_mem_mb").round(2),
        ])
        return self

    def sort_by_cuda(self):
        self.df = self.df.sort("cuda_self_ms", descending=True)
        return self

    def sort_by_cpu(self):
        self.df = self.df.sort("cpu_self_ms", descending=True)
        return self

    def show(self, n=20):
        print(self.df.head(n))
        return self

    def to_polars(self) -> pl.DataFrame:
        return self.df

    def to_csv(self, path):
        self.df.write_csv(path)
