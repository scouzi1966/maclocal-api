package metrics

import "strings"

func Key(name string, labels map[string]string) string {
    parts := []string{name}
    for k, v := range labels {
        parts = append(parts, k+"="+v)
    }
    return strings.Join(parts, ",")
}
