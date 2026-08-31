import QtQuick

// One AI model module, drawn as a small circular node.
//
// States: offline | idle | thinking | active | error | rate_limited.
// Every animation is bound to `nodeState`, so an idle JARVIS runs no
// animation at all beyond the very slow idle breath -- an offline node
// runs none whatsoever.
Item {
    id: node

    property string label: ""
    property string modelName: ""
    property string reason: ""
    property string nodeState: "offline"
    property bool available: false
    property real diameter: 74

    width: diameter
    height: diameter

    readonly property color colorOffline:  "#16222e"
    readonly property color colorIdle:     "#2f8fd8"
    readonly property color colorThinking: "#4ad0ff"
    readonly property color colorActive:   "#00e9ff"
    readonly property color colorError:    "#ff4d5e"
    // Amber, deliberately NOT the error red: a throttled module is
    // configured and working, just being told to wait.
    readonly property color colorLimited:  "#ffab2e"

    readonly property color accent:
          nodeState === "rate_limited" ? colorLimited
        : nodeState === "error"    ? colorError
        : nodeState === "active"   ? colorActive
        : nodeState === "thinking" ? colorThinking
        : nodeState === "idle"     ? colorIdle
        :                            colorOffline

    // How strongly the node is lit. Offline stays deliberately very dim.
    readonly property real intensity:
          nodeState === "rate_limited" ? 0.7
        : nodeState === "error"    ? 0.85
        : nodeState === "active"   ? 1.0
        : nodeState === "thinking" ? 0.8
        : nodeState === "idle"     ? 0.5
        :                            0.16

    // --- glow: three concentric translucent rings. Cheap, GPU friendly,
    // and it does not need a graphical-effects module (which is not part
    // of PySide6-Essentials).
    Repeater {
        model: 3
        Rectangle {
            required property int index
            anchors.centerIn: parent
            width: node.diameter + index * 11
            height: width
            radius: width / 2
            color: "transparent"
            border.color: node.accent
            border.width: index === 0 ? 1.6 : 1.0
            opacity: node.intensity * (index === 0 ? 0.95 : (index === 1 ? 0.22 : 0.10))
            // The state change itself is instant; the eye sees a smooth
            // transition instead of a jump.
            Behavior on opacity { NumberAnimation { duration: 320; easing.type: Easing.OutCubic } }
            Behavior on border.color { ColorAnimation { duration: 320 } }
        }
    }

    // Faint filled core so the node reads as a disc, not just an outline.
    Rectangle {
        anchors.centerIn: parent
        width: node.diameter - 6
        height: width
        radius: width / 2
        color: node.accent
        opacity: node.intensity * 0.10
        Behavior on opacity { NumberAnimation { duration: 320 } }
    }

    // --- THINKING: a segmented ring that rotates.
    Canvas {
        id: spinner
        anchors.centerIn: parent
        width: node.diameter + 16
        height: width
        visible: node.nodeState === "thinking"
        opacity: 0.85
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var r = width / 2 - 2;
            ctx.lineWidth = 2;
            ctx.strokeStyle = node.colorThinking;
            for (var i = 0; i < 3; ++i) {
                var start = i * (2 * Math.PI / 3);
                ctx.beginPath();
                ctx.arc(width / 2, height / 2, r, start, start + 0.72);
                ctx.stroke();
            }
        }
        RotationAnimator on rotation {
            running: spinner.visible
            loops: Animation.Infinite
            from: 0; to: 360
            duration: 2600
        }
    }

    // --- THINKING: a gentle brightness pulse layered on the glow.
    SequentialAnimation on opacity {
        running: node.nodeState === "thinking"
        loops: Animation.Infinite
        NumberAnimation { to: 0.72; duration: 780; easing.type: Easing.InOutSine }
        NumberAnimation { to: 1.0;  duration: 780; easing.type: Easing.InOutSine }
    }

    // --- ACTIVE: one short, smooth expansion pulse.
    Rectangle {
        id: activePulse
        anchors.centerIn: parent
        width: node.diameter
        height: width
        radius: width / 2
        color: "transparent"
        border.color: node.colorActive
        border.width: 1.4
        opacity: 0
        SequentialAnimation {
            running: node.nodeState === "active"
            loops: Animation.Infinite
            ParallelAnimation {
                NumberAnimation { target: activePulse; property: "scale"; from: 1.0; to: 1.5; duration: 900; easing.type: Easing.OutCubic }
                NumberAnimation { target: activePulse; property: "opacity"; from: 0.75; to: 0.0; duration: 900; easing.type: Easing.OutCubic }
            }
        }
    }

    // --- ERROR: a soft red halo, not a flash.
    Rectangle {
        anchors.centerIn: parent
        width: node.diameter + 18
        height: width
        radius: width / 2
        color: "transparent"
        // Uses the node's own accent so the SAME halo serves the red
        // error and the amber rate limit -- one animation, two meanings,
        // rather than a second near-identical element.
        border.color: node.accent
        border.width: 1
        visible: node.nodeState === "error" || node.nodeState === "rate_limited"
        SequentialAnimation on opacity {
            running: node.nodeState === "error" || node.nodeState === "rate_limited"
            loops: Animation.Infinite
            NumberAnimation { to: 0.42; duration: 1100; easing.type: Easing.InOutSine }
            NumberAnimation { to: 0.12; duration: 1100; easing.type: Easing.InOutSine }
        }
    }

    // --- labels
    Column {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.bottom
        anchors.topMargin: 8
        spacing: 2

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: node.label.toUpperCase()
            color: node.available ? "#cfeaff" : "#4a6070"
            font.pixelSize: 11
            font.letterSpacing: 1.6
            font.family: "Segoe UI"
            font.bold: true
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            // "RATE_LIMITED" reads better as "RATE LIMITED".
            text: node.available ? node.nodeState.toUpperCase().replace("_", " ")
                                 : (node.reason ? node.reason : "OFFLINE")
            color: node.nodeState === "error" ? "#ff8b96"
                 : node.nodeState === "rate_limited" ? "#ffc46a"
                 : (node.available ? "#4f93c4" : "#37505f")
            font.pixelSize: 9
            font.letterSpacing: 1.1
            font.family: "Segoe UI"
        }
    }
}
