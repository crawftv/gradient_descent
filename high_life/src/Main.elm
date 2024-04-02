port module Main exposing (..)

import Browser
import Html exposing (..)
import Html.Attributes exposing (style, value, wrap)
import Html.Events exposing (..)
import Http
import Json.Decode as Decode
import Json.Decode.Pipeline exposing (required)
import Json.Encode
import RemoteData exposing (RemoteData(..), WebData)
import RemoteData.Http


type alias Key =
    String


type alias Value =
    String



-- MAIN


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }


appName =
    "Crawford's Recommender"



-- MODEL


type alias Model =
    { input : Input
    , request : WebData LLMRequest
    , newRow : TableRow
    , tableRows : TableRows
    }


type alias ResponseText =
    String


type alias RequestText =
    String


type alias Input =
    String


type TableRow
    = NoInput
    | HasOnlyInput LLMRequest
    | HasAllData FullTableRow


type alias TableRows =
    List TableRow


type alias LLMRequest =
    { requestText : RequestText
    }


type alias LLMResponse =
    { responseText : ResponseText
    }


type alias FullTableRow =
    { requestText : RequestText, responseText : ResponseText }


init : Json.Encode.Value -> ( Model, Cmd Msg )
init flags =
    case Decode.decodeValue tableRowsDecoder flags of
        Ok tableRows ->
            ( Model "" NotAsked NoInput tableRows, Cmd.none )

        Err _ ->
            ( Model "" NotAsked NoInput [], Cmd.none )



--
-- UPDATE


type Msg
    = SendRequest String
    | UpdateRequest String
    | UseResponseToUpdateModel (WebData LLMResponse)
    | LoadTableRowsFromLocalStorage Json.Encode.Value
    | SaveTableRowsToLocalStorage
    | DeleteRow Int


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        SendRequest input ->
            case model.newRow of
                HasOnlyInput data ->
                    let
                        newRow =
                            HasOnlyInput data

                        tableRows =
                            newRow :: model.tableRows
                    in
                    ( { input = input, request = Loading, newRow = newRow, tableRows = tableRows }, postLLMRequest input )

                _ ->
                    ( { model | request = NotAsked }, Cmd.none )

        UpdateRequest newRequest ->
            ( { model | input = newRequest, newRow = HasOnlyInput (LLMRequest newRequest) }, Cmd.none )

        UseResponseToUpdateModel webDataResponse ->
            case webDataResponse of
                Success new_llmResponse ->
                    case model.newRow of
                        HasOnlyInput data ->
                            let
                                newRow =
                                    HasAllData { requestText = data.requestText, responseText = new_llmResponse.responseText }

                                tableRows =
                                    newRow :: List.drop 1 model.tableRows
                            in
                            ( { input = "", request = NotAsked, newRow = NoInput, tableRows = tableRows }
                            , saveTableRows <| encodeTableRows tableRows
                            )

                        _ ->
                            ( model, Cmd.none )

                NotAsked ->
                    ( model, Cmd.none )

                Failure err ->
                    ( { model
                        | request = Failure err
                        , input = model.input
                        , newRow = HasOnlyInput { requestText = model.input }
                        , tableRows = List.drop 1 model.tableRows
                      }
                    , Cmd.none
                    )

                Loading ->
                    ( { model | request = Loading }, Cmd.none )

        LoadTableRowsFromLocalStorage json ->
            case Decode.decodeValue tableRowsDecoder json of
                Ok tableRows ->
                    ( { model | tableRows = tableRows }, Cmd.none )

                Err _ ->
                    ( model, Cmd.none )

        SaveTableRowsToLocalStorage ->
            ( model, saveTableRows <| encodeTableRows model.tableRows )

        DeleteRow index ->
            let
                tableRows =
                    List.take index model.tableRows ++ List.drop (index + 1) model.tableRows
            in
            ( { model | tableRows = tableRows }, saveTableRows <| encodeTableRows tableRows )


subscriptions : Model -> Sub Msg
subscriptions _ =
    loadTableRows LoadTableRowsFromLocalStorage



-- VIEW


view : Model -> Html Msg
view model =
    div []
        [ h2 [] [ text appName ]
        , h3 [] [ text "Get recommendations on Food, Wine, Travel, and more from trusted sources." ]
        , viewApp model
        ]


viewInputCell : Input -> Html Msg
viewInputCell input =
    td [] [ text input ]


viewResponseCell : ResponseText -> Html Msg
viewResponseCell responseText =
    td []
        [ pre
            [ style "white-space" "pre-wrap"
            , style "font-family" "inherit"
            ]
            [ text responseText ]
        ]


viewTableRow : Int -> TableRow -> Maybe (Html Msg)
viewTableRow index tableRow =
    case tableRow of
        NoInput ->
            Nothing

        HasOnlyInput llmRequest ->
            Just <|
                tr []
                    [ viewInputCell llmRequest.requestText
                    , td [] [ progress [] [] ]
                    ]

        HasAllData data ->
            Just <|
                tr []
                    [ td []
                        [ button [ onClick (DeleteRow index) ]
                            [ text "Delete" ]
                        ]
                    , viewInputCell data.requestText
                    , viewResponseCell data.responseText
                    ]


viewTableRows : TableRows -> List (Maybe (Html Msg))
viewTableRows tableRows =
    List.indexedMap viewTableRow tableRows


maybeToList : Maybe a -> List a
maybeToList m =
    case m of
        Nothing ->
            []

        Just x ->
            [ x ]


filterRow : TableRow -> List FullTableRow
filterRow tableRow =
    case tableRow of
        HasAllData data ->
            [ data ]

        _ ->
            []


thStyling : List (Attribute msg)
thStyling =
    [ style "text-align" "left" ]


viewResponses : TableRows -> Html Msg
viewResponses tableRows =
    table []
        [ thead []
            [ th [] []
            , th thStyling [ text "Query" ]
            , th thStyling [ text "Answer" ]
            ]
        , tbody [] <| List.concatMap maybeToList (viewTableRows tableRows)
        ]


viewApp : Model -> Html Msg
viewApp model =
    let
        state =
            model.request
    in
    case state of
        NotAsked ->
            div []
                [ text "Try a search"
                , input [ onInput UpdateRequest, Html.Attributes.name <| appName ++ " Query" ] []
                , button [ onClick (SendRequest model.input) ] [ text "Ask!" ]
                , viewResponses model.tableRows
                ]

        Failure err ->
            div []
                [ decodeError model err
                , input [ onInput UpdateRequest, value model.input ] [ text model.input ]
                , button [ onClick (SendRequest model.input) ] [ text "Ask!" ]
                , viewResponses model.tableRows
                ]

        Loading ->
            div []
                [ viewResponses model.tableRows
                ]

        Success _ ->
            div []
                [ text "Try a search"
                , input [ onInput UpdateRequest ] []
                , button [ onClick <| SendRequest model.input ] [ text "Ask!" ]
                , viewResponses model.tableRows
                ]



-- HTTP


postLLMRequest : String -> Cmd Msg
postLLMRequest input =
    --  (crossOrigin "http://localhost:8001/query" [] [])
    RemoteData.Http.postWithConfig RemoteData.Http.defaultConfig "/query" UseResponseToUpdateModel llmResponseDecoder (encodeRequest input)


encodeRequest : String -> Json.Encode.Value
encodeRequest input =
    Json.Encode.object
        [ ( "text", Json.Encode.string input ) ]


encodeResponse : String -> Json.Encode.Value
encodeResponse input =
    Json.Encode.object
        [ ( "text", Json.Encode.string input ) ]


encodeTableRow : FullTableRow -> Json.Encode.Value
encodeTableRow row =
    Json.Encode.object
        [ ( "requestText", Json.Encode.string row.requestText ), ( "responseText", Json.Encode.string row.responseText ) ]


encodeTableRows : TableRows -> Json.Encode.Value
encodeTableRows rows =
    Json.Encode.list encodeTableRow <| List.concat <| List.map filterRow rows


llmResponseDecoder : Decode.Decoder LLMResponse
llmResponseDecoder =
    Decode.succeed LLMResponse
        |> required "text" Decode.string


decodeError : Model -> Http.Error -> Html Msg
decodeError model error =
    case error of
        Http.BadUrl string ->
            div []
                [ text string
                ]

        Http.Timeout ->
            div []
                [ text "Timeout :("
                ]

        Http.NetworkError ->
            div []
                [ text "Network Error :("
                ]

        Http.BadStatus int ->
            div []
                [ text ("Bad Status: " ++ String.fromInt int)
                ]

        Http.BadBody string ->
            div []
                [ text ("Bad Request Body :(" ++ " " ++ string)
                , viewResponses model.tableRows
                ]



-- PORTS


port loadTableRows : (Json.Encode.Value -> msg) -> Sub msg


port sendLoadTableRowsMsg : () -> Cmd msg


port saveTableRows : Json.Encode.Value -> Cmd msg


tableRowDecoder : Decode.Decoder TableRow
tableRowDecoder =
    Decode.map2 (\request response -> HasAllData { requestText = request, responseText = response })
        (Decode.field "requestText" Decode.string)
        (Decode.field "responseText" Decode.string)


tableRowsDecoder : Decode.Decoder TableRows
tableRowsDecoder =
    Decode.list tableRowDecoder
