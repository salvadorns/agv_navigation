#!/bin/bash
# Setup del Hotspot Wi-Fi en la Raspberry Pi 4B
# Ejecutar UNA SOLA VEZ como root, pasando la contraseña por variable de entorno:
#   sudo HOTSPOT_PASS='la_contraseña' bash setup_hotspot.sh
# (sudo descarta las variables exportadas, por eso va en la misma línea)

set -e

: "${HOTSPOT_PASS:?Definir HOTSPOT_PASS con la contraseña del hotspot (mín. 8 caracteres)}"

echo "==> Instalando dependencias..."
apt-get update -qq
apt-get install -y python3-flask python3-opencv networkmanager

echo "==> Creando Hotspot con NetworkManager..."
nmcli device wifi hotspot \
    ifname wlan0 \
    ssid "Cuadri_Net" \
    password "$HOTSPOT_PASS" \
    con-name "Cuadri_Hotspot"

# Fijar la IP del servidor en 10.42.0.1
nmcli connection modify "Cuadri_Hotspot" \
    ipv4.addresses 10.42.0.1/24 \
    ipv4.method shared

nmcli connection up "Cuadri_Hotspot"

echo "==> Creando servicio systemd para el servidor Flask..."
cat > /etc/systemd/system/telemetria.service << 'EOF'
[Unit]
Description=Servidor de Telemetria Cuatriciclo
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/admin/servidor_telemetria.py
WorkingDirectory=/home/admin
Restart=always
User=admin
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable telemetria.service
systemctl start telemetria.service

echo ""
echo "==> LISTO. Hotspot 'Cuadri_Net' activo en 10.42.0.1"
echo "==> Servidor Flask corriendo en puerto 5000"
echo ""
echo "Para probar:"
echo "  curl -X POST -d 'data=48_1500_1' http://localhost:5000/update"
