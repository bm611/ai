package main

import (
	"io"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"testing"
)

func BenchmarkCLIStartup(b *testing.B) {
	if _, err := os.Stat("bin/ai"); err != nil {
		b.Skip("build bin/ai before running the startup benchmark")
	}
	var maxRSS int64
	b.ResetTimer()
	for range b.N {
		cmd := exec.Command("./bin/ai", "config", "show")
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
		if err := cmd.Run(); err != nil {
			b.Fatal(err)
		}
		if rss := processMaxRSS(cmd); rss > maxRSS {
			maxRSS = rss
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(maxRSS), "B-maxrss")
}

func processMaxRSS(cmd *exec.Cmd) int64 {
	value := reflect.ValueOf(cmd.ProcessState.SysUsage())
	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return 0
	}
	field := value.FieldByName("Maxrss")
	if !field.IsValid() || !field.CanInt() {
		return 0
	}
	rss := field.Int()
	if runtime.GOOS != "darwin" {
		rss *= 1024
	}
	return rss
}
