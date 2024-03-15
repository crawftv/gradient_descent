module Main exposing (..)

import Browser
import Html exposing (..)
import Html.Attributes exposing (style)
import Html.Events exposing (..)
import Http
import Json.Decode as Decode exposing (Decoder, decodeString, float, int, nullable, string)
import Json.Decode.Pipeline exposing (hardcoded, optional, required)
import Json.Encode
import RemoteData exposing (RemoteData(..), WebData)
import RemoteData.Http
import Url.Builder exposing (crossOrigin)



-- MAIN


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type alias Model =
    { input : String
    , request : WebData LLMRequest
    , newRow : TableRow
    , tableRows : TableRows
    }


type TableRow
    = NoInput
    | HasOnlyInput LLMRequest
    | HasAllData { request : LLMRequest, response : LLMResponse }


type alias TableRows =
    List TableRow


type alias LLMRequest =
    { text : String
    }


type alias LLMResponse =
    { text : String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( Model "" NotAsked NoInput [], Cmd.none )



-- UPDATE


type Msg
    = SendRequest String
    | UpdateRequest String
    | UseResponseToUpdateModel (WebData LLMResponse)


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
                    ( { input = "", request = Loading, newRow = newRow, tableRows = tableRows }, postLLMRequest input )

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
                                    HasAllData { request = data, response = new_llmResponse }

                                tableRows =
                                    List.drop 1 model.tableRows
                            in
                            ( { input = "", request = NotAsked, newRow = NoInput, tableRows = newRow :: tableRows }, Cmd.none )

                        _ ->
                            ( model, Cmd.none )

                NotAsked ->
                    ( model, Cmd.none )

                Failure err ->
                    ( { model | request = Failure err, input = model.input, newRow = HasOnlyInput { text = model.input }, tableRows = List.drop 1 model.tableRows }, Cmd.none )

                Loading ->
                    ( { model | request = Loading }, Cmd.none )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.none



-- VIEW


view : Model -> Html Msg
view model =
    div []
        [ h2 []
            [ text "Recommender"
            , h3 [] [ text "Get recommendations on Food, Wine, Travel, and more from trusted sources." ]
            ]
        , viewQuote model
        ]


viewInputCell : LLMRequest -> Html Msg
viewInputCell llmRequest =
    td [] [ text llmRequest.text ]


viewResponseCell : LLMResponse -> Html Msg
viewResponseCell llmRequest =
    td [] [ text llmRequest.text ]


viewTableRow : TableRow -> Maybe (Html Msg)
viewTableRow tableRow =
    case tableRow of
        NoInput ->
            Nothing

        HasOnlyInput llmRequest ->
            Just <|
                tr []
                    [ viewInputCell llmRequest
                    , td [] [ text "Loading" ]
                    ]

        HasAllData data ->
            Just <|
                tr []
                    [ viewInputCell data.request
                    , viewResponseCell data.response
                    ]


viewTableRows : TableRows -> List (Maybe (Html Msg))
viewTableRows tableRows =
    List.map viewTableRow tableRows


maybeToList : Maybe a -> List a
maybeToList m =
    case m of
        Nothing ->
            []

        Just x ->
            [ x ]


viewResponses : TableRows -> Html Msg
viewResponses tableRows =
    table []
        [ thead []
            [ th [] [ text "Query" ]
            , th [] [ text "Answer" ]
            ]
        , tbody [] <| List.concatMap maybeToList (viewTableRows tableRows)
        ]


viewQuote : Model -> Html Msg
viewQuote model =
    let
        state =
            model.request
    in
    case state of
        NotAsked ->
            div []
                [ text "Try a search"
                , input [ onInput UpdateRequest ] []
                , button [ onClick (SendRequest model.input) ] [ text "Ask!" ]
                , viewResponses model.tableRows
                ]

        Failure err ->
            div []
                [ decodeError model err
                , input [ onInput UpdateRequest ] [ text model.input ]
                , button [ onClick (SendRequest model.input) ] [ text "Ask!" ]
                , viewResponses model.tableRows
                , text (Debug.toString model)
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


llmResponseDecoder : Decoder LLMResponse
llmResponseDecoder =
    Decode.succeed LLMResponse
        |> required "text" string


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
                [ text ("Bad Request Body :(" ++ " " ++ string) ]
