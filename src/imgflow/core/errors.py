class ImgflowError(Exception):
    """Tüm imgflow hatalarının ortak temeli."""


class CycleError(ImgflowError):
    """Graph'ta bir döngü tespit edildiğinde fırlatılır."""


class UnresolvedOperatorError(ImgflowError):
    """Registry'de bulunamayan bir operatör id'sine referans verildiğinde fırlatılır."""


class OperatorRuntimeError(ImgflowError):
    """Bir operatörün run() çağrısı sırasında oluşan hatayı sarmalar."""

    def __init__(self, node_id: str, op_id: str, original: Exception):
        self.node_id = node_id
        self.op_id = op_id
        self.original = original
        super().__init__(f"Node '{node_id}' ({op_id}) çalıştırılırken hata: {original}")


class RecipeVersionError(ImgflowError):
    """Reçete dosyası, uygulamanın anlayabileceğinden daha yeni bir şema sürümündeyse fırlatılır."""
