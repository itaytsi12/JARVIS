import QtQuick
import QtQuick.Window
import "components"

// The JARVIS AI core window.
//
// Pure black, no side panels, one centred core. Everything on screen is
// driven by `bridge` (ui/ui_bridge.py) -- there is no simulated state
// anywhere in this file.
Window {
    id: root

    // `startFullscreen` is injected from Python (ui/app.py) out of the
    // project's own configuration, so the window mode is a setting, not a
    // hard-coded choice, and Escape always gets you out of it.
    property bool startFullscreen: false

    width: 1280
    height: 800
    minimumWidth: 640
    minimumHeight: 480
    visible: true
    color: "#000000"
    title: "JARVIS"
    visibility: startFullscreen ? Window.FullScreen : Window.Maximized

    // Where each model node sits on the ring, in degrees clockwise from
    // 12 o'clock. Laid out to match the requested arrangement:
    //
    //                 OpenAI
    //         Gemini          Anthropic
    //           Local LLM       Vision
    readonly property var nodeAngles: ({
        "openai":    0,
        "anthropic": 68,
        "vision":    132,
        "local":     228,
        "gemini":    292
    })

    function angleFor(id) {
        return nodeAngles[id] !== undefined ? nodeAngles[id] : 0;
    }

    Item {
        anchors.fill: parent
        focus: true

        // Escape always leaves fullscreen -- never trapped.
        Keys.onEscapePressed: {
            if (root.visibility === Window.FullScreen)
                root.visibility = Window.Maximized;
        }
        Keys.onPressed: function (event) {
            if (event.key === Qt.Key_F11) {
                root.visibility = (root.visibility === Window.FullScreen)
                    ? Window.Maximized : Window.FullScreen;
                event.accepted = true;
            }
        }

        // ---------------------------------------------------------------
        // Header
        // ---------------------------------------------------------------
        Column {
            id: header
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.top
            anchors.topMargin: Math.max(28, parent.height * 0.05)
            spacing: 8

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "JARVIS AI CORE"
                color: "#dff2ff"
                font.family: "Segoe UI"
                font.pixelSize: Math.max(20, root.height * 0.032)
                font.letterSpacing: Math.max(6, root.height * 0.009)
                font.bold: true
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                // The REAL number of configured model modules, from
                // ui/model_status.py. Never a hard-coded 5.
                text: bridge.subtitle
                color: "#3f9ad0"
                font.family: "Segoe UI"
                font.pixelSize: Math.max(11, root.height * 0.016)
                font.letterSpacing: 4
            }
        }

        // ---------------------------------------------------------------
        // The core
        // ---------------------------------------------------------------
        // The core gets the whole band between the header and the footer,
        // and never more: it is sized from the space actually left over, so
        // the ring can never grow across "N MODELS ONLINE" or
        // "ACTIVE | LEARNING | ADAPTING" on a short window.
        Item {
            id: stage

            readonly property real bandTop: header.y + header.height + 18
            readonly property real bandBottom: footer.y - 18
            readonly property real side: Math.max(240, Math.min(parent.width - 60, bandBottom - bandTop))

            width: side
            height: side
            x: (parent.width - width) / 2
            y: bandTop + (bandBottom - bandTop - height) / 2

            readonly property real cx: width / 2
            readonly property real cy: height / 2
            // Distance from the hub to each model node.
            readonly property real orbit: width * 0.31

            CoreRing {
                anchors.fill: parent
                listening: bridge.listening
                speaking: bridge.speaking
                processing: bridge.processing
                ready: bridge.ready
            }

            // Connection lines: hub -> every model node. Drawn first so
            // the nodes sit on top of them.
            Repeater {
                model: bridge.models
                ConnectionLine {
                    required property var modelData
                    readonly property real ang: root.angleFor(modelData.id) * Math.PI / 180
                    readonly property bool lit: modelData.state === "thinking" || modelData.state === "active"

                    x1: stage.cx
                    y1: stage.cy
                    x2: stage.cx + Math.sin(ang) * stage.orbit
                    y2: stage.cy - Math.cos(ang) * stage.orbit
                    accent: modelData.state === "error" ? "#ff4d5e"
                          : modelData.state === "rate_limited" ? "#ffab2e"
                          : lit ? "#00d8ff"
                          : (modelData.available ? "#2f8fd8" : "#16222e")
                    intensity: modelData.available ? (lit ? 0.85 : 0.30) : 0.10
                    flowing: lit
                }
            }

            Repeater {
                model: bridge.models
                ModelNode {
                    required property var modelData
                    readonly property real ang: root.angleFor(modelData.id) * Math.PI / 180

                    diameter: Math.max(54, stage.width * 0.115)
                    label: modelData.label
                    modelName: modelData.modelName
                    reason: modelData.reason
                    nodeState: modelData.state
                    available: modelData.available
                    x: stage.cx + Math.sin(ang) * stage.orbit - width / 2
                    y: stage.cy - Math.cos(ang) * stage.orbit - height / 2
                }
            }
        }

        // ---------------------------------------------------------------
        // Footer
        // ---------------------------------------------------------------
        Column {
            id: footer
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: Math.max(24, parent.height * 0.045)
            spacing: 10
            width: Math.min(parent.width * 0.8, 900)

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "ACTIVE  |  LEARNING  |  ADAPTING"
                color: "#2f8fd8"
                font.family: "Segoe UI"
                font.pixelSize: Math.max(11, root.height * 0.0155)
                font.letterSpacing: 5
                opacity: 0.85
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: bridge.statusText
                color: "#5aa9d8"
                font.family: "Segoe UI"
                font.pixelSize: 11
                font.letterSpacing: 2
                opacity: 0.7
                elide: Text.ElideRight
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
            }

            // The last thing heard and the last thing said. Present so the
            // bridge's set_user_text/set_jarvis_text are actually visible,
            // and empty (not placeholder text) when nothing has happened.
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                visible: bridge.userText.length > 0
                text: "“" + bridge.userText + "”"
                color: "#7fd4ff"
                font.family: "Segoe UI"
                font.pixelSize: 13
                elide: Text.ElideRight
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                visible: bridge.jarvisText.length > 0
                text: bridge.jarvisText
                color: "#bfe6ff"
                font.family: "Segoe UI"
                font.pixelSize: 13
                elide: Text.ElideRight
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
                opacity: 0.9
            }
        }
    }
}
