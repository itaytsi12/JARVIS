import QtQuick

// The central JARVIS AI core: concentric rings, segmented arcs, radial
// ticks, sparse circuitry and a small central hub.
//
// Performance note: every piece of detail here is painted into a Canvas
// exactly once (on resize). Motion is done with RotationAnimator and
// opacity animations, which run on the scene-graph render thread and
// never trigger a repaint. So the whole HUD costs one texture upload at
// startup and pure transforms afterwards.
Item {
    id: core

    // Real backend state, forwarded from the bridge.
    property bool listening: false
    property bool speaking: false
    property bool processing: false
    property bool ready: false

    readonly property real radius: Math.min(width, height) / 2
    readonly property color blue: "#2f8fd8"
    readonly property color cyan: "#00d8ff"
    readonly property color faint: "#12384f"

    // ---------------------------------------------------------------
    // Outer glow ring -- the large circle everything else sits inside.
    // ---------------------------------------------------------------
    Repeater {
        model: 4
        Rectangle {
            required property int index
            anchors.centerIn: parent
            width: core.radius * 2 - index * 7
            height: width
            radius: width / 2
            color: "transparent"
            border.color: index === 0 ? core.cyan : core.blue
            border.width: index === 0 ? 1.8 : 1
            opacity: index === 0 ? 0.55 : (0.20 / index)
        }
    }

    // A very faint filled disc so the core reads as a volume against the
    // pure black background.
    Rectangle {
        anchors.centerIn: parent
        width: core.radius * 2
        height: width
        radius: width / 2
        color: core.blue
        opacity: 0.022
    }

    // Slow "breathing" of the outer ring while JARVIS is listening.
    Rectangle {
        id: listenHalo
        anchors.centerIn: parent
        width: core.radius * 2 + 10
        height: width
        radius: width / 2
        color: "transparent"
        border.color: core.cyan
        border.width: 1.2
        opacity: 0
        SequentialAnimation on opacity {
            running: core.listening
            loops: Animation.Infinite
            NumberAnimation { to: 0.45; duration: 900; easing.type: Easing.InOutSine }
            NumberAnimation { to: 0.06; duration: 900; easing.type: Easing.InOutSine }
        }
    }

    // ---------------------------------------------------------------
    // Segmented arc ring -- rotates slowly, clockwise.
    // ---------------------------------------------------------------
    Canvas {
        id: arcs
        anchors.centerIn: parent
        width: core.radius * 2 - 26
        height: width
        onWidthChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var r = width / 2 - 2;
            var segments = [
                [0.00, 0.34], [0.42, 0.14], [0.62, 0.42], [1.12, 0.10],
                [1.30, 0.28], [1.72, 0.36]
            ];
            ctx.lineWidth = 2.4;
            ctx.strokeStyle = core.cyan;
            ctx.globalAlpha = 0.55;
            for (var i = 0; i < segments.length; ++i) {
                var start = segments[i][0] * Math.PI;
                ctx.beginPath();
                ctx.arc(width / 2, height / 2, r, start, start + segments[i][1] * Math.PI);
                ctx.stroke();
            }
        }
        RotationAnimator on rotation {
            loops: Animation.Infinite
            from: 0; to: 360
            duration: 48000
        }
    }

    // A second, thinner arc ring rotating the other way.
    Canvas {
        id: arcsInner
        anchors.centerIn: parent
        width: core.radius * 2 - 62
        height: width
        onWidthChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var r = width / 2 - 2;
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = core.blue;
            ctx.globalAlpha = 0.5;
            for (var i = 0; i < 4; ++i) {
                var start = i * Math.PI / 2 + 0.18;
                ctx.beginPath();
                ctx.arc(width / 2, height / 2, r, start, start + Math.PI / 2 - 0.36);
                ctx.stroke();
            }
        }
        RotationAnimator on rotation {
            loops: Animation.Infinite
            from: 360; to: 0
            duration: 72000
        }
    }

    // ---------------------------------------------------------------
    // Radial tick marks -- static, every 5 degrees, longer every 30.
    // ---------------------------------------------------------------
    Canvas {
        id: ticks
        anchors.centerIn: parent
        width: core.radius * 2 - 12
        height: width
        onWidthChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var cx = width / 2, cy = height / 2, r = width / 2 - 2;
            for (var deg = 0; deg < 360; deg += 5) {
                var major = (deg % 30) === 0;
                var len = major ? 9 : 4;
                var a = deg * Math.PI / 180;
                ctx.beginPath();
                ctx.globalAlpha = major ? 0.55 : 0.24;
                ctx.strokeStyle = major ? core.cyan : core.blue;
                ctx.lineWidth = major ? 1.4 : 1;
                ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
                ctx.lineTo(cx + Math.cos(a) * (r - len), cy + Math.sin(a) * (r - len));
                ctx.stroke();
            }
        }
    }

    // ---------------------------------------------------------------
    // Circuitry detail -- a sparse, deterministic set of traces between
    // two inner radii. Deterministic (no Math.random) so the HUD looks
    // identical on every launch instead of reshuffling.
    // ---------------------------------------------------------------
    Canvas {
        id: circuitry
        anchors.centerIn: parent
        width: core.radius * 2 - 96
        height: width
        onWidthChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var cx = width / 2, cy = height / 2;
            var outer = width / 2 - 4;
            var inner = outer * 0.72;
            ctx.strokeStyle = core.blue;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.30;
            var seeds = [12, 47, 88, 131, 168, 203, 241, 288, 316, 344];
            for (var i = 0; i < seeds.length; ++i) {
                var a = seeds[i] * Math.PI / 180;
                var step = ((i % 3) + 1) * 0.045;
                var x1 = cx + Math.cos(a) * outer, y1 = cy + Math.sin(a) * outer;
                var x2 = cx + Math.cos(a) * (inner + 6), y2 = cy + Math.sin(a) * (inner + 6);
                var x3 = cx + Math.cos(a + step) * inner, y3 = cy + Math.sin(a + step) * inner;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.lineTo(x3, y3);
                ctx.stroke();
                // A small terminal dot where the trace ends.
                ctx.beginPath();
                ctx.globalAlpha = 0.55;
                ctx.fillStyle = core.cyan;
                ctx.arc(x3, y3, 1.5, 0, 2 * Math.PI);
                ctx.fill();
                ctx.globalAlpha = 0.30;
            }
        }
    }

    // ---------------------------------------------------------------
    // Sparse particles -- 14 slow light points. Deliberately not a
    // particle SYSTEM: fourteen animated Rectangles is a rounding error,
    // a real emitter would not be.
    // ---------------------------------------------------------------
    Repeater {
        model: 14
        Rectangle {
            required property int index
            readonly property real ang: index * (360 / 14) * Math.PI / 180
            readonly property real dist: core.radius * (0.30 + ((index * 37) % 60) / 100.0)
            width: 2
            height: 2
            radius: 1
            color: (index % 3 === 0) ? "#ffffff" : core.cyan
            x: core.width / 2 + Math.cos(ang) * dist - 1
            y: core.height / 2 + Math.sin(ang) * dist - 1
            SequentialAnimation on opacity {
                loops: Animation.Infinite
                NumberAnimation { to: 0.65; duration: 1800 + (index % 5) * 700; easing.type: Easing.InOutSine }
                NumberAnimation { to: 0.06; duration: 2200 + (index % 4) * 600; easing.type: Easing.InOutSine }
            }
        }
    }

    // ---------------------------------------------------------------
    // Central hub. Pulses softly while listening, more strongly while
    // speaking, and holds a steady brighter state while processing.
    // ---------------------------------------------------------------
    Item {
        id: hub
        anchors.centerIn: parent
        width: 46
        height: 46

        readonly property color hubColor: core.speaking ? core.cyan
                                        : core.processing ? "#4ad0ff"
                                        : core.blue

        Repeater {
            model: 3
            Rectangle {
                required property int index
                anchors.centerIn: parent
                width: hub.width - index * 12
                height: width
                radius: width / 2
                color: "transparent"
                border.color: hub.hubColor
                border.width: index === 2 ? 2 : 1
                opacity: core.ready ? (0.85 - index * 0.2) : 0.25
                Behavior on opacity { NumberAnimation { duration: 400 } }
            }
        }

        Rectangle {
            anchors.centerIn: parent
            width: 8
            height: 8
            radius: 4
            color: hub.hubColor
            opacity: 0.95
        }

        // Listening: a soft outward ripple.
        Rectangle {
            id: listenRipple
            anchors.centerIn: parent
            width: hub.width
            height: width
            radius: width / 2
            color: "transparent"
            border.color: core.cyan
            border.width: 1
            opacity: 0
            SequentialAnimation {
                running: core.listening && !core.speaking
                loops: Animation.Infinite
                ParallelAnimation {
                    NumberAnimation { target: listenRipple; property: "scale"; from: 1.0; to: 2.4; duration: 1600; easing.type: Easing.OutCubic }
                    NumberAnimation { target: listenRipple; property: "opacity"; from: 0.5; to: 0.0; duration: 1600; easing.type: Easing.OutCubic }
                }
            }
        }

        // Speaking: a faster, stronger ripple on the core itself.
        Rectangle {
            id: speakRipple
            anchors.centerIn: parent
            width: hub.width
            height: width
            radius: width / 2
            color: "transparent"
            border.color: core.cyan
            border.width: 1.6
            opacity: 0
            SequentialAnimation {
                running: core.speaking
                loops: Animation.Infinite
                ParallelAnimation {
                    NumberAnimation { target: speakRipple; property: "scale"; from: 1.0; to: 3.4; duration: 900; easing.type: Easing.OutCubic }
                    NumberAnimation { target: speakRipple; property: "opacity"; from: 0.8; to: 0.0; duration: 900; easing.type: Easing.OutCubic }
                }
            }
        }
    }
}
